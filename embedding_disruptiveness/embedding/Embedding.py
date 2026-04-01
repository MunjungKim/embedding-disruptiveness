# -*- coding: utf-8 -*-
# @Author: Munjung Kim
# @Date:   2024-10-31 14:33:12

import os
import logging
import numpy as np
import scipy.sparse
import torch
import embedding_disruptiveness
import tqdm

class EmbeddingTrainer:
    def __init__(
        self,
        net_input,
        dim,
        window_size,
        device_in,
        device_out,
        q_value,
        epochs,
        batch_size,
        save_dir,
        num_walks=25,
        walk_length=160,
        learning_rate=1e-3,
        num_workers=10,
        dtype_reduce = False
    ):
        """
        A class to train embedding space of a citation network.

        Parameters:
            net_input (str or scipy.sparse.csr_matrix): Path to the citation network file (.npz) or a CSR matrix.
            dim (int): Dimension of the embedding vectors.
            window_size (int): Size of the context window.
            device_in (str): Device ID for in-vectors.
            device_out (str): Device ID for out-vectors and training.
            q_value (float): Value of q for the biased random walk.
            epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.
            save_dir (str): Directory to save the embeddings.
            num_walks (int): Number of walks per node (default: 25).
            walk_length (int): Length of each walk (default: 160).
            learning_rate (float): Learning rate for training (default: 1e-3).
            num_workers (int): Number of worker threads (default: 10).
        """
        self.net_input = net_input
        self.dim = dim
        self.window_size = window_size
        self.device_in = device_in
        self.device_out = device_out
        self.q_value = q_value
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.dtype_reduce = dtype_reduce
        
        self.in_vec = None
        self.out_vec = None
        self.embedding_disruptiveness = None

        self._setup_devices()
        self._load_network()
        self._prepare_model()
        
    def _setup_devices(self):
        # Normalize device strings: accept both '0' and 'cuda:0'
        if not str(self.device_in).startswith('cuda'):
            self.device_in = f'cuda:{self.device_in}'
        if not str(self.device_out).startswith('cuda'):
            self.device_out = f'cuda:{self.device_out}'
        logging.info("Devices: %s (in), %s (out)", self.device_in, self.device_out)


    def _load_network(self):
        logging.info("Loading network...")
        if isinstance(self.net_input, str):
            # If net_input is a file path, load the network from the .npz file
            self.net = scipy.sparse.load_npz(self.net_input)
            logging.info(f"Network loaded from file: {self.net_input}")
        elif isinstance(self.net_input, scipy.sparse.csr_matrix):
            # If net_input is a CSR matrix, use it directly
            self.net = self.net_input
            logging.info("Network loaded from CSR matrix input.")
        else:
            raise ValueError(
                "net_input must be either a file path (str) or a scipy.sparse.csr_matrix."
            )
        self.n_nodes = self.net.shape[0]

    def _prepare_model(self):
        logging.info("Preparing model...")
        self.model = embedding_disruptiveness.Word2Vec(
            vocab_size=self.n_nodes,
            embedding_size=self.dim,
            padding_idx=self.n_nodes,
            device_in=self.device_in,
            device_out=self.device_out,
            dtype_reduce=self.dtype_reduce,
        )
        self.loss_func = embedding_disruptiveness.Node2VecTripletLoss(n_neg=1)

    def _prepare_dataset(self):
        logging.info("Preparing dataset...")
        sampler = embedding_disruptiveness.RandomWalkSampler(
            self.net, walk_length=self.walk_length
        )
        noise_sampler = embedding_disruptiveness.utils.node_sampler.ConfigModelNodeSampler(
            ns_exponent=1.0
        )
        noise_sampler.fit(self.net)

        self.dataset = embedding_disruptiveness.TripletDataset(
            adjmat=self.net,
            window_length=self.window_size,
            num_walks=self.num_walks,
            noise_sampler=noise_sampler,
            padding_id=self.n_nodes,
            buffer_size=1e4,
            context_window_type="right",
            epochs=self.epochs,
            negative=1,
            p=1,
            q=self.q_value,
            walk_length=self.walk_length,
        )

    def train(self):
        self._prepare_dataset()
        logging.info("Starting training...")

        embedding_disruptiveness.train(
            model=self.model,
            dataset=self.dataset,
            loss_func=self.loss_func,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_workers=self.num_workers,
            dtype_reduce = self.dtype_reduce
        )
        logging.info("Training completed.")
        self.model.eval()
        dtype = torch.float32
        
        if self.dtype_reduce:
            self.in_vec = self.model.ivectors.weight.to(dtype=dtype).cpu().detach().numpy()[: self.n_nodes, :]
            self.out_vec = self.model.ovectors.weight.to(dtype=dtype).cpu().detach().numpy()[: self.n_nodes, :]
        else:
            self.in_vec = self.model.ivectors.weight.data.cpu().detach().numpy()[: self.n_nodes, :]
            self.out_vec = self.model.ovectors.weight.data.cpu().detach().numpy()[: self.n_nodes, :]

    def save_embeddings(self):
        print("Saving embeddings...")
        
        os.makedirs(self.save_dir, exist_ok=True)
        np.save(os.path.join(self.save_dir, "in.npy"), self.in_vec)
        np.save(os.path.join(self.save_dir, "out.npy"), self.out_vec)
        print(f"Embeddings saved to {self.save_dir}")
        
    def cal_embedding_disruptiveness(self):
        print("Calculating embedding disruptiveness...")
        self.in_vec = self.model.ivectors.weight.data.cpu().to(dtype=torch.float32)[: self.n_nodes, :]
        self.out_vec = self.model.ovectors.weight.data.cpu().to(dtype=torch.float32)[: self.n_nodes, :]

        n = len(self.out_vec)

        distance= []
        
        batch_size = max(1, int(n / 100) + 1)
        n_batches = (n + batch_size - 1) // batch_size

        logging.info('Starting calculating the distances')

        for i in tqdm.tqdm(range(n_batches)):
            X = self.in_vec[i*batch_size: (i+1)*batch_size]
            Y = self.out_vec[i*batch_size: (i+1)*batch_size]
            numerator = torch.sum(X * Y, dim=1)
            norms_X = torch.sqrt((X * X).sum(axis=1))
            norms_Y = torch.sqrt((Y * Y).sum(axis=1))

            denominator = norms_X*norms_Y


            cs = 1 - torch.divide(numerator, denominator)
            distance.append(cs.tolist())
        
        self.embedding_disruptiveness =  np.array([dis for  sublist in distance for dis in sublist])
        
        logging.info('Saving the files.')
        
        SAVE_DIR = os.path.join(self.save_dir,'distance.npy')
        np.save(SAVE_DIR,self.embedding_disruptiveness)