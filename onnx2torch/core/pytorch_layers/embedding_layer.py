import torch
import torch.nn as nn


class Embedding(nn.Module):
    @classmethod
    def from_onnx(cls, mod):
        weight = nn.Parameter(
            torch.from_numpy(mod.inputs[0].values), requires_grad=False
        )
        num_embeddings, embedding_dim = weight.shape
        embedding = nn.Embedding(num_embeddings, embedding_dim)
        embedding.weight = weight

        return embedding
