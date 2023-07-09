import torch
import numpy as np
import tiktoken
from transformers import GPT2LMHeadModel

tokenizer = tiktoken.get_encoding("gpt2")


def get_single_encoding(cfg, pos):
    return [np.sin(pos / np.power(10000, i / cfg.embedding_dimension)) if i % 2 == 0 else np.cos(
        pos / np.power(10000, (i - 1) / cfg.embedding_dimension)) for i in range(cfg.embedding_dimension)]


def get_positional_encoding(cfg):
    return torch.FloatTensor([get_single_encoding(cfg, i) for i in range(cfg.context_length)]).unsqueeze(0)
# def get_single_encoding(cfg, pos):
#     return [np.sin(pos / np.power(10000, i / cfg.n_embd)) if i % 2 == 0 else np.cos(
#         pos / np.power(10000, (i - 1) / cfg.n_embd)) for i in range(cfg.n_embd)]


# def get_positional_encoding(cfg):
#     return torch.FloatTensor([get_single_encoding(cfg, i) for i in range(cfg.block_size)]).unsqueeze(0)


def encode(text):
    return tokenizer.encode(text)


def decode(tokens):
    return tokenizer.decode(tokens)


def data_prep(input_file='Shakespeare_input.txt', output_file='shakespeare_tokenized.pt'):
    with open(input_file, 'r') as file:
        inp = file.read()
    data_tensor = torch.tensor([int(s) for s in tokenizer.encode(inp)])  # .split('\n')

    # with open('Shakespeare_tokenized', 'w') as file: #only used for manually viewing the tokens
    #    file.write('\n'.join([str(s) for s in data_tensor]))

    torch.save(data_tensor, output_file)
    print("Dataset has been tokenized into ", len(data_tensor), " tokens")

def load_embedding_weights():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    we = model.transformer.wte.weight
    torch.save(we, 'token_embedding_weights.pt')
    print("Weights of shape", we.shape, "have been loaded")