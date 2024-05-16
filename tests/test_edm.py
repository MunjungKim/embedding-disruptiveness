import scipy
import torch
import importlib
import sys
import edmsci

NET = "/data/sg/munjkim/APS/original/citation_net.npz"

net = scipy.sparse.load_npz(NET)

sampler = edmsci.RandomWalkSampler(net, walk_length = 60)

n_nodes = net.shape[0]

model =edmsci.Word2Vec(vocab_size = n_nodes, embedding_size = 100)

model.multi_gpus(devices = ["0","1"])

noise_sampler = edmsci.ConfigModelNodeSampler(ns_exponent = 1.0)
noise_sampler.fit(net)

dataset = edmsci.TripletDataset(adjmat= net, window_length = 3, num_walks = 25, noise_sampler = noise_sampler, padding_id = n_nodes, buffer_size = 134, context_window_type = "right", epochs=10, negative=1, p=1, q=1, walk_length=160)

model.fit(dataset = dataset,batch_size = 1024,num_workers= 10)

model.eval()

in_vec = model.invectors.weight.data.cpu().numpuy()[:n_nodes,:]
out_vec = model.ovectors.weight.data.cpu().numpuy()[:n_nodes,:]

np.save("temp_in.npy", in_vec)

np.save("temp_out.npy", out_vec)