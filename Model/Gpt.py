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
        self.task = cfg.task

        self.token_embedding = nn.Embedding(cfg.vocab_size,
                                            cfg.embedding_dimension)

        if cfg.embedding_dimension == 768:
            print("Using pre-trained Embedding weights from gpt2")
            self.token_embedding.weight = torch.load('token_embedding_weights.pt')
            self.token_embedding.weight.requires_grad = False

        self.register_buffer('positional_embedding', Embeddings.get_positional_encoding(cfg))
        # self.positional_layer = nn.Embedding(cfg.context_length, cfg.embedding_dimension)
        self.embedding_dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_blocks)])
        self.norm = nn.LayerNorm(cfg.embedding_dimension)

        if self.task == "prediction":
            self.head = nn.Linear(cfg.embedding_dimension, cfg.vocab_size, bias=False)
            self.head.weight = self.token_embedding.weight  # should be no_grad through weight tying
        else:
            out_dim = 1 if self.task == "stsb" else 3 if self.task.startswith("mnli") else 2
            self.ft_head = nn.Linear(cfg.embedding_dimension, out_dim)
            # Different Name is used, so load_state_dict ignores the old language modelling head (strict=False)

    def forward(self, x):
        freezeModel = True
        if freezeModel:
            self.eval()
        with torch.set_grad_enabled(not freezeModel):
            x = self.token_embedding(x) + self.positional_embedding[:, :x.shape[1], :]
            # x = self.token_embedding(x) + self.positional_layer(torch.arange(0, x.shape[1], dtype=torch.long, device=x.device))
            x = self.embedding_dropout(x)

            x = self.norm(self.blocks(x))

        if not self.task == "prediction": # When Finetuning only the last position in context_dim is forwarded to the head, because it holds the ful information
            return self.ft_head(x[:, -1:, :])
        return self.head(x)

    def generate(self, x, max_length, temperature=1.0):
        for i in range(max_length):
            logits = self(x.view(1, -1)[:, -self.T:])
            logits = logits[:, -1, :] / temperature  # .view(-1)
            probs = func.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)  # torch.argmax(probs)
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
        # self.context_length = T
        self.n_embd = C
        self.n_head = nh
        self.scale_factor = 1 / math.sqrt(hs)
        self.dropout = cfg.dropout

        self.c_attn = nn.Linear(C, 3 * C, bias=False)
        # stdv = 1 / torch.sqrt(torch.tensor(C))  # (num_heads, C, head_size)#
        # self.W_q = nn.Parameter(nn.init.uniform_(torch.empty(nh, C, hs), a=-stdv, b=stdv))
        # self.W_k = nn.Parameter(nn.init.uniform_(torch.empty(nh, C, hs), a=-stdv, b=stdv))
        # self.W_v = nn.Parameter(nn.init.uniform_(torch.empty(nh, C, hs), a=-stdv, b=stdv))

        self.linear = nn.Linear(C, C)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer('low_tri', torch.tril(torch.ones(T, T)).view(1, 1, T, T))
            print('Manual attention implementation will be used')
        # else:
        # print('using torch sdp')
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # with torch.cuda.amp.autocast_mode.autocast(enabled=False):
        # shape(x): B, T, C (parallel, context, embedding)
        B, T, C = x.shape
        # x = x[:, None, :, :]  # Introduce Dummy dimension to fit the heads dimension of weights

        # Q = torch.matmul(x, self.W_q)  # (B, num_heads, T, head_size)
        # K = torch.matmul(x, self.W_k)
        # V = torch.matmul(x, self.W_v)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        K = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        Q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        V = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            result = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=None,
                                                                      dropout_p=self.dropout if self.training else 0,
                                                                      is_causal=True)
        else:
            score = torch.matmul(Q, K.transpose(-2,
                                                -1)) * self.scale_factor  # / Q.size(-1) ** 0.5#self.context_length ** 0.5
            weight = score.masked_fill(self.low_tri[:, :, :T, :T] == 0, float('-inf'))
            soft_score = torch.softmax(weight, dim=-1)
            result = torch.matmul(soft_score, V)  # shape: (B, num_head T, head_size)

        res = result.transpose(1, 2).reshape(B, T, C)
        res = self.linear(res)
        res = self.drop(res)
        return res