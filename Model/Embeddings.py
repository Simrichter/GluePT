import torch
import numpy as np
import tiktoken
# from transformers import GPT2LMHeadModel

# This function returns an encoding vector for a given position.
# It is used as a helper function to create the positional embedding matrix and to extend it if needed in the GPT model.
def get_single_encoding(embedding_dimension, pos):
    return [np.sin(pos / np.power(10000, i / embedding_dimension)) if i % 2 == 0 else np.cos(
        pos / np.power(10000, (i - 1) / embedding_dimension)) for i in range(embedding_dimension)]

# This function creates a positional embedding matrix of shape "cfg.context_length" X "cfg.embedding_dimension" (T X C)
def get_positional_encoding(cfg):
    return torch.FloatTensor([get_single_encoding(cfg.embedding_dimension, i) for i in range(cfg.context_length)]).unsqueeze(0)



# For tokenization we use byte pair encoding with a vocabulary already pretrained by OpenAI for their GPT2 models.
# Therefore we also use OpenAI's library for encoding and decoding
tokenizer = tiktoken.get_encoding("gpt2")

def encode(text):
    return tokenizer.encode(text)

def decode(tokens):
    return tokenizer.decode(tokens)


# def data_prep(input_file='Shakespeare_input.txt', output_file='shakespeare_tokenized.pt'):
#     with open(input_file, 'r') as file:
#         inp = file.read()
#     data_tensor = torch.tensor([int(s) for s in tokenizer.encode(inp)])  # .split('\n')

#     # with open('Shakespeare_tokenized', 'w') as file: #only used for manually viewing the tokens
#     #    file.write('\n'.join([str(s) for s in data_tensor]))

#     torch.save(data_tensor, output_file)
#     print("Dataset has been tokenized into ", len(data_tensor), " tokens")

# def load_embedding_weights():
#     model = GPT2LMHeadModel.from_pretrained('gpt2')
#     we = model.transformer.wte.weight
#     torch.save(we, 'token_embedding_weights.pt')
#     print("Weights of shape", we.shape, "have been loaded")