import torch
import numpy as np
import tiktoken

# This function returns an encoding vector for a given position.
# It is used as a helper function to create the positional embedding matrix and to extend it if needed in the GPT model.
def get_single_encoding(embedding_dimension, pos):
    return [np.sin(pos / np.power(10000, i / embedding_dimension)) if i % 2 == 0 else np.cos(
        pos / np.power(10000, (i - 1) / embedding_dimension)) for i in range(embedding_dimension)]

# This function creates a positional embedding matrix of shape "cfg.context_length" X "cfg.embedding_dimension" (T X C)
def get_positional_encoding(cfg):
    return torch.FloatTensor([get_single_encoding(cfg.embedding_dimension, i) for i in range(cfg.context_length)]).unsqueeze(0)


# For tokenization we use byte pair encoding with a vocabulary already pretrained by OpenAI for their GPT2 models.
# Therefore we also use OpenAI's tiktoken library for encoding and decoding
tokenizer = tiktoken.get_encoding("gpt2")

def encode(text):
    return tokenizer.encode(text)

def decode(tokens):
    return tokenizer.decode(tokens)