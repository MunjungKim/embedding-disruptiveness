# embedding-disruptiveness

`embedding-disruptiveness` is a Python package for calculating the Embedding Disruptiveness Index of papers or patents using a citation network. This measure helps to identify how disruptive or consolidating a publication or patent is within its respective field.

The package builds upon the [node2vec](https://github.com/skojaku/node2vec) code by @skojaku.


## Installation

To install the latest version of `embedding-disruptiveness`, run:

```bash
pip install --upgrade embedding-disruptiveness
```


## Requirements
This code requires at least two gpus

## Usage
Here is a basic example of how to use embedding-disruptiveness:

```python
import embedding_disruptiveness

# Initialize the model with required parameters
trainer = embedding_disruptiveness.EmbeddingTrainer(net_input = NETWORK_FILE_LOCATION , #(npz file type)
                                                   dim = 128, # dimension of embedding vectors
                                                   window_size=5, # windowsize
                                                   device_in = '6', # cuda device where in-vectors will be 
                                                   device_out = '7',# cuda device where out-vectors will be 
                                                   q_value = 1, # q value in the randomwalk
                                                   epochs =1,  # epochsize
                                                    batch_size = 1024,# batchsize
                                                   save_dir = SAVE_LOCATION)

trainer.train()
```

## Model Parallelism

This package uses model parallelism to speed up the calculations, especially when dealing with large citation networks. Note: You need at least two GPUs to run the code. Ensure your environment is set up for multi-GPU usage, and the necessary CUDA drivers are installed.