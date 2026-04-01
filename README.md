# embedding-disruptiveness

A Python package for measuring how disruptive a paper or patent is, using graph embeddings on citation networks.

`embedding-disruptiveness` learns node2vec-style embeddings from citation graphs and computes an *Embedding Disruptiveness Measure (EDM)* that captures whether a work disrupts or consolidates its field. It also provides the classic *Disruption Index (DI)* as a built-in utility.

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/embedding-disruptiveness.svg)](https://pypi.org/project/embedding-disruptiveness/)

[**Paper**](https://arxiv.org/abs/2502.16845) | [**Blog Post**](https://munjungkim.github.io/embedding-disruptiveness-blog/) | [**PyPI**](https://pypi.org/project/embedding-disruptiveness/)

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Input Format](#input-format)
- [Training Embeddings](#training-embeddings)
- [Computing the Disruption Index](#computing-the-disruption-index)
- [Negative Samplers](#negative-samplers)
- [Custom Training Loop](#custom-training-loop)
- [API Reference](#api-reference)
- [How It Works](#how-it-works)
- [Citation](#citation)
- [License](#license)

---

## Installation

### Using pip

```bash
pip install embedding-disruptiveness
```

### Using uv

```bash
uv pip install embedding-disruptiveness
```

### Install from source

```bash
git clone https://github.com/MunjungKim/embedding-disruptiveness.git
cd embedding-disruptiveness
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch (with CUDA for GPU training)
- NumPy, SciPy, scikit-learn, numba, tqdm

## Quick Start

```python
import embedding_disruptiveness as edm

# Train embeddings on a citation network (scipy sparse matrix in .npz format)
trainer = edm.EmbeddingTrainer(
    net_input="citation_network.npz",
    dim=128,
    window_size=5,
    device_in="0",       # GPU for in-vectors
    device_out="1",      # GPU for out-vectors
    q_value=1,
    epochs=5,
    batch_size=1024,
    save_dir="./output",
)

trainer.train()
# Embeddings and cosine distances are saved to save_dir
```

For a detailed walkthrough of the method and results, see the [blog post](https://munjungkim.github.io/embedding-disruptiveness-blog/).

## Input Format

Your citation network should be a **scipy sparse matrix** saved as `.npz`. Rows and columns represent nodes (papers or patents), and non-zero entries represent citation edges.

```python
import scipy.sparse as sp

net = sp.csr_matrix(adjacency_data)
sp.save_npz("citation_network.npz", net)
```

You can also convert an edge list to a sparse adjacency matrix:

```python
import numpy as np
from embedding_disruptiveness.utils import to_adjacency_matrix

# (N, 2) edge list: [src, dst]
edges = np.array([[0, 1], [1, 2], [2, 3]])
net = to_adjacency_matrix(edges, edgelist=True)

# (N, 3) weighted edge list: [src, dst, weight]
weighted_edges = np.array([[0, 1, 0.5], [1, 2, 1.0], [2, 3, 0.8]])
net = to_adjacency_matrix(weighted_edges, edgelist=True)
```

## Training Embeddings

`EmbeddingTrainer` handles the full pipeline — loading the network, generating biased random walks, training a Word2Vec-style model with triplet loss, and saving the resulting embeddings.

```python
trainer = edm.EmbeddingTrainer(
    net_input="network.npz",
    dim=128,               # Embedding dimension
    window_size=5,         # Context window for skip-gram
    device_in="0",         # CUDA device for in-vectors
    device_out="1",        # CUDA device for out-vectors
    q_value=1,             # Node2Vec return parameter (q)
    epochs=5,              # Training epochs
    batch_size=1024,       # Batch size
    save_dir="./results",  # Output directory
    num_walks=10,          # Random walks per node (default: 10)
    walk_length=80,        # Walk length (default: 80)
)

trainer.train()
```

## Computing the Disruption Index

The package also provides standalone functions for computing the disruption index directly from a citation network, without training embeddings.

```python
import embedding_disruptiveness as edm

# 1-step disruption index
di = edm.calc_disruption_index(net)

# 2-step (multistep) disruption index
di_2step = edm.calc_multistep_disruption_index(net)
```

Three computation methods are available via the `method` parameter:

- `"matrix"` — sparse matrix multiplication. Fast for small networks (< 1M nodes), but uses more memory.
- `"iterative"` — Numba-JIT row-wise loop. Memory-efficient, scales to 100M+ nodes.
- `"auto"` (default) — picks `"matrix"` for networks under 1M nodes, `"iterative"` otherwise.

```python
# Force iterative method for a large network
di = edm.calc_disruption_index(large_net, method="iterative")

# Force matrix method with batching for medium networks
di = edm.calc_disruption_index(net, method="matrix", batch_size=2**15)
```

## Negative Samplers

Different null models yield different notions of "expected" connections. You can choose from several built-in samplers:

```python
from embedding_disruptiveness.utils import (
    ConfigModelNodeSampler,
    SBMNodeSampler,
    ErdosRenyiNodeSampler,
)

# Configuration Model — preserves degree distribution
sampler = ConfigModelNodeSampler(adj_matrix)

# Stochastic Block Model — preserves community structure
sampler = SBMNodeSampler(adj_matrix, group_membership)

# Erdos-Renyi — uniform random baseline
sampler = ErdosRenyiNodeSampler(adj_matrix)
```

## Custom Training Loop

If you need more control, you can use the individual components directly:

```python
from embedding_disruptiveness.models import Word2Vec
from embedding_disruptiveness.loss import Node2VecTripletLoss
from embedding_disruptiveness.datasets import TripletDataset
from embedding_disruptiveness.torch import train

model = Word2Vec(
    vocab_size=num_nodes, embedding_size=128,
    padding_idx=num_nodes, device_in="cuda:0", device_out="cuda:1"
)
dataset = TripletDataset(center, context, negative_sampler, epochs=5)
loss_fn = Node2VecTripletLoss()

train(model=model, dataset=dataset, loss_func=loss_fn, batch_size=1024)
```

## API Reference

| Module | Key Exports | Description |
|--------|-------------|-------------|
| `embedding` | `EmbeddingTrainer` | High-level training orchestrator |
| `models` | `Word2Vec` | Dual-device embedding model |
| `loss` | `Node2VecTripletLoss`, `ModularityTripletLoss` | Loss functions |
| `datasets` | `TripletDataset`, `ModularityDataset` | PyTorch datasets for triplet sampling |
| `torch` | `train()` | Training loop with AMP and SparseAdam |
| `utils` | `RandomWalkSampler`, `*NodeSampler`, `calc_disruption_index`, `calc_multistep_disruption_index` | Samplers and metrics |

## How It Works

1. **Random Walks** — Node2Vec-style biased walks explore the citation graph, capturing both local and global structure via (p, q) parameters.

2. **Directional Skip-Gram** — A Word2Vec model learns separate *in-vectors* (as a target) and *out-vectors* (as a context) for each node, preserving the directionality of citations.

3. **Embedding Disruptiveness** — Cosine distances between a focal paper's in-vector and its references'/citations' out-vectors quantify how much the paper departs from, or reinforces, existing knowledge.

For more details, see our [paper](https://arxiv.org/abs/2502.16845) and [blog post](https://munjungkim.github.io/embedding-disruptiveness-blog/).

## Citation

If you use this package in your research, please cite:

```bibtex
@article{kim2024embedding,
  title={Embedding Disruptiveness Measure},
  author={Kim, Munjung and others},
  year={2024}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/MunjungKim/embedding-disruptiveness).
