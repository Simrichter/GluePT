import torch
import torch.nn as nn
from torch.nn import functional as func
import Model.Embeddings as Embeddings
import math


class GptModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device
        self.T = cfg.context_length

        self.token_embedding = nn.Embedding(cfg.vocab_size,
                                            cfg.embedding_dimension)

        if cfg.embedding_dimension == 768:
            print("Using pre-trained Embedding weights from gpt2")
            self.token_embedding.weight = torch.load('token_embedding_weights.pt')
            self.token_embedding.weight.requires_grad = False

        self.register_buffer('positional_embedding', Embeddings.get_positional_encoding(cfg))
        # self.positional_layer = nn.Embedding(cfg.context_length, cfg.embedding_dimension)
        self.embedding_dropout = nn.Dropout(cfg.dropout)

        self.emb_to_vocab = nn.Linear(cfg.embedding_dimension, cfg.vocab_size, bias=False)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_blocks)])
        self.norm = nn.LayerNorm(cfg.embedding_dimension)

        self.emb_to_vocab.weight = self.token_embedding.weight  # should be no_grad through weight tying
        # self.token_embedding.weight = self.emb_to_vocab.weight  # weight tying attempt #TODO check if this works

    def forward(self, x, y=None):
        x = self.token_embedding(x) + self.positional_embedding[:, :x.shape[1], :]
        # x = self.token_embedding(x) + self.positional_layer(torch.arange(0, x.shape[1], dtype=torch.long, device=x.device))
        x = self.embedding_dropout(x)

        x = self.norm(self.blocks(x))
        x = self.emb_to_vocab(x)

        if y is None:
            loss = None
        else:
            B, T, vs = x.shape
            # x = x.view(B * T, vs)
            # targets = y.view(B * T)
            loss = func.cross_entropy(x.view(B * T, vs), y.view(B * T), ignore_index=-1)
        return x, loss

    def generate(self, x, max_length, temperature=1.0):
        for i in range(max_length):
            logits, loss = self(x.view(1, -1)[:, -self.T:])
            logits = logits[:, -1, :] / temperature  # .view(-1)
            probs = func.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1) #torch.argmax(probs)
            x = torch.cat((x, x_next.reshape(1)))
        return x


class Gelu(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class Block(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        # self.sa = Multi_head_attention(cfg)
        self.sa = Multi_head_attention(cfg)
        self.norm1 = nn.LayerNorm(cfg.embedding_dimension)
        self.norm2 = nn.LayerNorm(cfg.embedding_dimension)
        self.ffwd = nn.Sequential(
            nn.Linear(cfg.embedding_dimension, 4 * cfg.embedding_dimension),
            Gelu(),
            nn.Linear(4 * cfg.embedding_dimension, cfg.embedding_dimension),
            nn.Dropout(cfg.dropout)
        )

    def forward(self, x):
        x = x + self.norm1(self.sa(x))  # TODO: Norm before Attention??
        x = x + self.norm2(self.ffwd(x))
        return x


class Multi_head_attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # B, T,
        T, C = cfg.context_length, cfg.embedding_dimension
        nh, hs = cfg.num_heads, C // cfg.num_heads
        self.context_length = T
        self.dropout = cfg.dropout

        stdv = 1 / torch.sqrt(torch.tensor(C))  # (num_heads, C, head_size)#
        self.W_q = nn.Parameter(nn.init.uniform_(torch.empty(nh, C, hs), a=-stdv, b=stdv))
        self.W_k = nn.Parameter(nn.init.uniform_(torch.empty(nh, C, hs), a=-stdv, b=stdv))
        self.W_v = nn.Parameter(nn.init.uniform_(torch.empty(nh, C, hs), a=-stdv, b=stdv))

        self.linear = nn.Linear(C, C)

        self.register_buffer('low_tri', torch.tril(torch.ones(T, T)).view(1, 1, T, T))

        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # shape(x): B, T, C (parallel, context, embedding)
        B, T, C = x.shape
        x = x[:, None, :, :]  # Introduce Dummy dimension to fit the heads dimension of weights

        Q = torch.matmul(x, self.W_q)  # (B, num_heads, T, head_size)
        K = torch.matmul(x, self.W_k)
        V = torch.matmul(x, self.W_v)

        score = torch.matmul(Q, K.transpose(-2, -1)) / self.context_length ** 0.5
        weight = score.masked_fill(self.low_tri[:, :, :T, :T] == 0, float('-inf'))
        soft_score = torch.softmax(weight, dim=-1)
        result = torch.matmul(soft_score, V)  # shape: (B, num_head T, head_size)

        res = result.transpose(1, 2).reshape(B, T, C)
        res = self.linear(res)
        res = self.drop(res)
        return res