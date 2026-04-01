import numpy as np
from scipy import sparse
from numba import njit, prange
from tqdm import tqdm


# ===========================================================================
# Public API
# ===========================================================================

def calc_disruption_index(net, method="auto", batch_size=None):
    """Calculate the disruption index (1-step).

    DI = (NF - NB) / (NR + NB + NF)

    Parameters
    ----------
    net : scipy.sparse matrix
        Citation network. net[i,j] = 1 if i cites j.
    method : str, default "auto"
        - "matrix" : sparse matrix multiplication (fast for small networks, OOM for large ones).
        - "iterative" : Numba-JIT row-wise loop (memory-efficient, scales to 100M+ nodes).
        - "auto" : "matrix" if n_nodes < 1_000_000, else "iterative".
    batch_size : int or None
        Only used with method="matrix". Splits computation into batches.

    Returns
    -------
    DI : np.ndarray of shape (n_nodes,)

    Reference
    ---------
    - Funk, R. J. & Owen-Smith, J. A dynamic network measure of technological change. Manage. Sci. 63, 791-817 (2017).
    - Wu, L., Wang, D. & Evans, J.A. Large teams develop and small teams disrupt science and technology. Nature 566, 378-382 (2019).
    """
    net = sparse.csr_matrix(net)
    net.setdiag(0)
    net.eliminate_zeros()
    n_nodes = net.shape[0]

    if method == "auto":
        method = "matrix" if n_nodes < 1_000_000 else "iterative"

    if method == "iterative":
        ref_indptr, ref_indices, cite_indptr, cite_indices = _build_csr_pair(net)
        return _calc_disruption_iterative(
            ref_indptr, ref_indices, cite_indptr, cite_indices, n_nodes
        )
    elif method == "matrix":
        if batch_size is None:
            return _calc_disruption_matrix(net)
        return _calc_disruption_matrix_batched(net, batch_size)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Choose 'auto', 'matrix', or 'iterative'."
        )


def calc_multistep_disruption_index(net, method="auto"):
    """Calculate the 2-step disruption index.

    Same as disruption index but references and citations are expanded
    to include 2-hop neighbors (references-of-references, citations-of-citations).

    Parameters
    ----------
    net : scipy.sparse matrix
        Citation network. net[i,j] = 1 if i cites j.
    method : str, default "auto"
        - "matrix" : sparse matrix multiplication.
        - "iterative" : Numba-JIT row-wise loop.
        - "auto" : "matrix" if n_nodes < 1_000_000, else "iterative".

    Returns
    -------
    DI : np.ndarray of shape (n_nodes,)
    """
    net = sparse.csr_matrix(net)
    net.setdiag(0)
    net.eliminate_zeros()
    n_nodes = net.shape[0]

    if method == "auto":
        method = "matrix" if n_nodes < 1_000_000 else "iterative"

    if method == "iterative":
        ref_indptr, ref_indices, cite_indptr, cite_indices = _build_csr_pair(net)
        return _calc_multistep_disruption_iterative(
            ref_indptr, ref_indices, cite_indptr, cite_indices, n_nodes
        )
    elif method == "matrix":
        return _calc_multistep_disruption_matrix(net)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Choose 'auto', 'matrix', or 'iterative'."
        )


# ===========================================================================
# Shared helper: build CSR pair (references + citations)
# ===========================================================================

def _build_csr_pair(net):
    """Extract CSR arrays for net (references) and net.T (citations)."""
    net = sparse.csr_matrix(net)
    net.data = np.ones_like(net.data)
    netT = sparse.csr_matrix(net.T)

    return (
        net.indptr.astype(np.int64),
        net.indices.astype(np.int64),
        netT.indptr.astype(np.int64),
        netT.indices.astype(np.int64),
    )


# ===========================================================================
# 1-step: matrix approach (original)
# ===========================================================================

def _calc_disruption_matrix(net):
    net = sparse.csr_matrix(net)
    net.data = net.data * 0 + 1

    AAT = net @ net.T
    AAT.data = np.ones_like(AAT.data)
    AAT.setdiag(0)
    AAT.eliminate_zeros()

    AT = sparse.csr_matrix(net.copy().T)
    AT.data = np.ones_like(AT.data)

    NB = AT.multiply(AAT)
    NF = AT - NB
    NR = AAT - NB

    DI = (NF.sum(axis=1) - NB.sum(axis=1)) / np.maximum(
        NR.sum(axis=1) + NB.sum(axis=1) + NF.sum(axis=1), 1
    )
    return np.asarray(DI).reshape(-1)


def _calc_disruption_matrix_batched(net, batch_size):
    net = sparse.csr_matrix(net)
    n_nodes = net.shape[0]
    n_chunks = max(1, int(n_nodes / batch_size))
    chunks = np.array_split(np.arange(n_nodes, dtype=int), n_chunks)
    DI = np.zeros(n_nodes)
    netT = sparse.csr_matrix(net.T)

    for focal_node_ids in tqdm(chunks):
        is_relevant = (
            np.array(net[focal_node_ids, :].sum(axis=0)).reshape(-1)
            + np.array(net[:, focal_node_ids].sum(axis=1)).reshape(-1)
            + np.array((net[focal_node_ids, :] @ netT).sum(axis=0)).reshape(-1)
        )
        is_relevant[focal_node_ids] = -1
        supp_node_ids = np.where(is_relevant > 0)[0]
        node_ids = np.concatenate([focal_node_ids, supp_node_ids])
        subnet = net[node_ids, :][:, node_ids].copy()
        subnet.sort_indices()
        dindex = _calc_disruption_matrix(subnet)
        DI[focal_node_ids] = dindex[: len(focal_node_ids)]
    return DI


# ===========================================================================
# 1-step: iterative approach (Numba, for large networks)
# ===========================================================================

@njit(parallel=True, cache=True)
def _calc_disruption_iterative(ref_indptr, ref_indices, cite_indptr, cite_indices, n):
    """Numba-JIT 1-step disruption index.

    net[i,j]=1 means i cites j:
      - references of i = ref_indices[ref_indptr[i]:ref_indptr[i+1]]
      - citations of i  = cite_indices[cite_indptr[i]:cite_indptr[i+1]]
    """
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        refs_start = ref_indptr[i]
        refs_end = ref_indptr[i + 1]
        cites_start = cite_indptr[i]
        cites_end = cite_indptr[i + 1]

        if refs_start == refs_end and cites_start == cites_end:
            continue

        # Build set of references of i
        refs_set = set()
        for ri in range(refs_start, refs_end):
            refs_set.add(ref_indices[ri])

        # NF, NB: iterate over papers that cite i
        n_f = 0
        n_b = 0
        cites_set = set()
        for ci in range(cites_start, cites_end):
            c = cite_indices[ci]
            cites_set.add(c)
            # Does c cite any reference of i?
            shared = False
            for cri in range(ref_indptr[c], ref_indptr[c + 1]):
                if ref_indices[cri] in refs_set:
                    shared = True
                    break
            if shared:
                n_b += 1
            else:
                n_f += 1

        # NR: papers not citing i but citing at least one reference of i
        n_k_set = set()
        for ri in range(refs_start, refs_end):
            ref_node = ref_indices[ri]
            for rci in range(cite_indptr[ref_node], cite_indptr[ref_node + 1]):
                candidate = cite_indices[rci]
                if candidate != i and candidate not in cites_set:
                    n_k_set.add(candidate)
        n_r = len(n_k_set)

        denom = n_f + n_b + n_r
        if denom > 0:
            result[i] = (n_f - n_b) / denom

    return result


# ===========================================================================
# 2-step (multistep): matrix approach
# ===========================================================================

def _calc_multistep_disruption_matrix(net):
    net = sparse.csr_matrix(net)
    net.data = net.data * 0 + 1

    # Expanded references (2-hop): refs(i) + refs(refs(i))
    net2 = net + net @ net
    net2.data = np.ones_like(net2.data)
    net2.setdiag(0)
    net2.eliminate_zeros()

    # Expanded citations (2-hop): cites(i) + cites(cites(i))
    netT = sparse.csr_matrix(net.T)
    netT2 = netT + netT @ netT
    netT2.data = np.ones_like(netT2.data)
    netT2.setdiag(0)
    netT2.eliminate_zeros()

    # AAT[i,j] = 1 if j's 1-hop refs overlap with i's expanded refs
    AAT = net2 @ net.T
    AAT.data = np.ones_like(AAT.data)
    AAT.setdiag(0)
    AAT.eliminate_zeros()

    # AT = expanded citations
    AT = netT2

    NB = AT.multiply(AAT)
    NF = AT - NB
    NR = AAT - NB

    DI = (NF.sum(axis=1) - NB.sum(axis=1)) / np.maximum(
        NR.sum(axis=1) + NB.sum(axis=1) + NF.sum(axis=1), 1
    )
    return np.asarray(DI).reshape(-1)


# ===========================================================================
# 2-step (multistep): iterative approach (Numba, for large networks)
# ===========================================================================

@njit(parallel=True, cache=True)
def _calc_multistep_disruption_iterative(
    ref_indptr, ref_indices, cite_indptr, cite_indices, n
):
    """Numba-JIT 2-step disruption index.

    For each node i:
      - refs_expanded  = refs(i) + refs(refs(i))
      - cites_expanded = cites(i) + cites(cites(i))
    Then compute DI using expanded sets.
    """
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        refs_start = ref_indptr[i]
        refs_end = ref_indptr[i + 1]
        cites_start = cite_indptr[i]
        cites_end = cite_indptr[i + 1]

        if refs_start == refs_end and cites_start == cites_end:
            continue

        # Build expanded references (2-hop)
        refs_set = set()
        for ri in range(refs_start, refs_end):
            r = ref_indices[ri]
            refs_set.add(r)
            for rri in range(ref_indptr[r], ref_indptr[r + 1]):
                refs_set.add(ref_indices[rri])
        refs_set.discard(i)

        # Build expanded citations (2-hop)
        cites_set = set()
        for ci in range(cites_start, cites_end):
            c = cite_indices[ci]
            cites_set.add(c)
            for cci in range(cite_indptr[c], cite_indptr[c + 1]):
                cites_set.add(cite_indices[cci])
        cites_set.discard(i)

        # NF, NB
        n_f = 0
        n_b = 0
        for c in cites_set:
            shared = False
            for cri in range(ref_indptr[c], ref_indptr[c + 1]):
                if ref_indices[cri] in refs_set:
                    shared = True
                    break
            if shared:
                n_b += 1
            else:
                n_f += 1

        # NR
        n_k_set = set()
        for r in refs_set:
            for rci in range(cite_indptr[r], cite_indptr[r + 1]):
                candidate = cite_indices[rci]
                if candidate != i and candidate not in cites_set:
                    n_k_set.add(candidate)
        n_r = len(n_k_set)

        denom = n_f + n_b + n_r
        if denom > 0:
            result[i] = (n_f - n_b) / denom

    return result
