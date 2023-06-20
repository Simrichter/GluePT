import torch
import torch.nn as nn
from torch.nn import functional as func
import Model.Embeddings as Embeddings


class GptModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device
        self.T = cfg.context_length
        self.token_embedding = nn.Embedding(cfg.vocab_size,
                                            cfg.embedding_dimension)  # TODO load pretrained weights from gpt2
        self.register_buffer('positional_embedding', Embeddings.get_positional_encoding(cfg))
        self.emb_to_vocab = nn.Linear(cfg.embedding_dimension, cfg.vocab_size, bias=False)
        # self.blocks = nn.ModuleList(*[Block(params) for head_id in range(params.n_blocks)])
        self.blocks = nn.Sequential(*[Block(cfg) for head_id in range(cfg.n_blocks)])
        self.norm = nn.LayerNorm(cfg.embedding_dimension)

        self.token_embedding.weight = self.emb_to_vocab.weight  # weight tying attempt #TODO check if this works

    def forward(self, x, y=None):
        x = self.token_embedding(x) + self.positional_embedding[:,x.shape[1]-1, :]
        x = self.norm(self.blocks(x))
        x = self.emb_to_vocab(x)

        if y is None:
            loss = None
        else:
            B, T, C = x.shape
            x = x.view(B * T, C)
            targets = y.view(B * T)
            loss = func.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1), ignore_index=-1)
        return x, loss

    def generate(self, x, max_length):
        for i in range(max_length):
            logits, loss = self(x[:, -self.T:])
            logits = logits[:, -1, :]
            probs = func.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            # print(x_next)
            x = torch.cat((x, x_next), dim=1)  # TODO Cropping to max context_width
        return x


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.sa = Multi_head_attention(cfg)
        self.norm1 = nn.LayerNorm(cfg.embedding_dimension)
        self.norm2 = nn.LayerNorm(cfg.embedding_dimension)
        self.ffwd = nn.Sequential(
            nn.Linear(cfg.embedding_dimension, 4 * cfg.embedding_dimension),
            nn.ReLU(),
            nn.Linear(4 * cfg.embedding_dimension, cfg.embedding_dimension)
        )

    def forward(self, x):
        x = x + self.norm1(self.sa(x))  # TODO: Norm before Attention??
        x = x + self.norm2(self.ffwd(x))
        return x


class Multi_head_attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #B, T,
        T, C = cfg.context_length, cfg.embedding_dimension
        nh, hs = cfg.num_heads, C // cfg.num_heads
        self.context_length = T

        stdv = 1 / torch.sqrt(torch.tensor(C))  # (num_heads, C, head_size)#
        self.W_q = nn.Parameter(nn.init.uniform_(torch.empty(nh, C, hs), a=-stdv, b=stdv))
        self.W_k = nn.Parameter(nn.init.uniform_(torch.empty(nh, C, hs), a=-stdv, b=stdv))
        self.W_v = nn.Parameter(nn.init.uniform_(torch.empty(nh, C, hs), a=-stdv, b=stdv))

        self.register_buffer('low_tri', torch.tril(torch.ones(T, T)).view(1, 1, T, T))
        self.linear = nn.Linear(cfg.embedding_dimension, cfg.embedding_dimension)

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
        return res
