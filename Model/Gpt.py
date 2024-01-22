import torch
import torch.nn as nn
from torch.nn import functional as func
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config
import Model.Embeddings as Embeddings
import math
import os


class GptModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.use_gpt2 = cfg.use_gpt2 #False#True#
        self.freeze_model = cfg.freeze_model
        self.device = cfg.device
        self.T = cfg.context_length
        self.task = cfg.task

        
        if self.use_gpt2:
            conf = GPT2Config(n_positions = cfg.context_length, n_embd = cfg.embedding_dimension, n_layer=cfg.n_blocks, n_head=cfg.num_heads, resid_pdrop=cfg.dropout, embd_pdrop=cfg.dropout, attn_pdrop=cfg.dropout) # Translates the config into a GPT2 config
            
            self.gpt2 = GPT2LMHeadModel(conf) if self.task == "prediction" else GPT2LMHeadModel.from_pretrained(f"gpt2{'-medium' if cfg.embedding_dimension == 1024 else ''}")
            self.train()
            
            print("GPT2 model loaded")
        else:
            self.train()
            self.token_embedding = nn.Embedding(cfg.vocab_size,
                                                cfg.embedding_dimension)

            if cfg.embedding_dimension == 768:
                print("Using pre-trained Embedding weights from gpt2")
                if not os.path.exists("Model/token_embedding_weights.pt"):
                    print("Downloading token_embedding_weights from gpt2")
                    tew = GPT2Model.from_pretrained("gpt2").wte.weight
                    torch.save(tew, "Model/token_embedding_weights.pt")
                self.token_embedding.weight = torch.load('Model/token_embedding_weights.pt')
                self.token_embedding.weight.requires_grad = False
            elif cfg.embedding_dimension == 1024:
                print("Using pre-trained Embedding weights from gpt2 Medium")
                if not os.path.exists("Model/token_embedding_weights_medium.pt"):
                    print("Downloading token_embedding_weights from gpt2 Medium")
                    tew = GPT2Model.from_pretrained("gpt2-medium").wte.weight
                    torch.save(tew, "Model/token_embedding_weights_medium.pt")
                self.token_embedding.weight = torch.load('Model/token_embedding_weights_medium.pt')
                self.token_embedding.weight.requires_grad = False

            self.register_buffer('positional_embedding', Embeddings.get_positional_encoding(cfg))
            # self.positional_layer = nn.Embedding(cfg.context_length, cfg.embedding_dimension)
            self.embedding_dropout = nn.Dropout(cfg.dropout)

            self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_blocks)])
            self.norm = nn.LayerNorm(cfg.embedding_dimension)

        if self.task == "prediction":
            self.head = nn.Linear(cfg.embedding_dimension, cfg.vocab_size, bias=False)
            if self.use_gpt2:
                self.head.weight = self.gpt2.transformer.wte.weight
            else:
                self.head.weight = self.token_embedding.weight  # should be no_grad through weight tying
        else:
            out_dim = 1 if self.task == "stsb" else 3 if "mnli" in self.task else 2
            self.ft_head = nn.Linear(cfg.embedding_dimension, out_dim) # Different Name is used for the head, so load_state_dict ignores the old language modelling head with (strict=False)
            self.ft_head.weight = torch.nn.Parameter(self.ft_head.weight/2) #TODO check this mod
            self.ft_head.bias = torch.nn.Parameter(self.ft_head.bias/2)
            

    def forward(self, x):
        if self.use_gpt2:
            x = self.gpt2(x, output_hidden_states=True).hidden_states[-1] # .last_hidden_state
        else:
            if self.freeze_model:
                self.eval()
            with torch.set_grad_enabled(not self.freeze_model):
                x = self.token_embedding(x) + self.positional_embedding[:, :x.shape[1], :]
                # x = self.token_embedding(x) + self.positional_layer(torch.arange(0, x.shape[1], dtype=torch.long, device=x.device))
                x = self.embedding_dropout(x)

                x = self.norm(self.blocks(x))
            self.train()


        if not self.task == "prediction": # When Finetuning only the last position in context_dim is forwarded to the head, because it holds the ful information
            #print(f"shape of x is: {x[:, -1:, :].shape}")
            out = self.ft_head(x[:, -1:, :])
            return out
        return self.head(x)#x.logits#

    def generate(self, x, max_length, temperature=1.0):
        for i in range(max_length):
            logits = self(x.view(1, -1)[:, -self.T:])
            #print(logits.shape)
            #logits = logits[:, -1, :] / temperature  # .view(-1) TODO is temperature needed?
            logits = logits.view(-1, 50257)
            probs = func.softmax(logits, dim=-1)
            print(probs.shape)
            x_next = torch.multinomial(probs, num_samples=1)  # torch.argmax(probs)
            print([Embeddings.decode(x_n.tolist()) for x_n in x_next])
            x_next = x_next[-1]
            x = torch.cat((x, x_next.reshape(1)))
        return x


class Block(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        # self.sa = Multi_head_attention(cfg)
        self.sa = Multi_head_attention(cfg)
        self.norm1 = nn.LayerNorm(cfg.embedding_dimension)
        self.norm2 = nn.LayerNorm(cfg.embedding_dimension)
        self.ffwd = nn.Sequential(
            nn.Linear(cfg.embedding_dimension, 4 * cfg.embedding_dimension),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * cfg.embedding_dimension, cfg.embedding_dimension),
            nn.Dropout(cfg.dropout)
        )
        # As described in "Language Models are Unsupervised Multitask Learners" all residual Layers (last layer before the residual path gets added)
        # are scaled by 1/sqrt(N) after initialization with stdv=0.02 (see "Improving Language Understanding by Generative Pre-Training")
        # since there are 2 residual connections per block, N equals 2*cfg.n_blocks
        torch.nn.init.normal_(self.ffwd[2].weight, 0, 0.02 / math.sqrt(2 * cfg.n_blocks))

    def forward(self, x):
        x = x + self.sa(self.norm1(x))
        x = x + self.ffwd(self.norm2(x))
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
        #self.bias = cfg.bias

        self.c_attn = nn.Linear(C, 3 * C, bias=cfg.bias) # TODO set Bias True (or not? not really common)
        # stdv = 1 / torch.sqrt(torch.tensor(C)) # Used to imitate the standard initialization of a linear layer
        # self.W_q = nn.Parameter(nn.init.uniform_(torch.empty(nh, C, hs), a=-stdv, b=stdv))
        # self.W_k = nn.Parameter(nn.init.uniform_(torch.empty(nh, C, hs), a=-stdv, b=stdv))
        # self.W_v = nn.Parameter(nn.init.uniform_(torch.empty(nh, C, hs), a=-stdv, b=stdv))
        #if cfg.bias:
        #    self.B_q = nn.Parameter

        self.linear = nn.Linear(C, C)
        # scaled initialization of residual layers (same as above)
        torch.nn.init.normal_(self.linear.weight, 0, 0.02/math.sqrt(2*cfg.n_blocks))

        self.sdp = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.sdp:
            # causal mask to ensure that attention is only applied to the left in the input sequence TODO ist der satz meiner?
            self.register_buffer('low_tri', torch.tril(torch.ones(T, T)).view(1, 1, T, T))
            print('Manual attention implementation will be used, might produce NaNs in combination with AMP')
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

        if self.sdp:
            # using the torch implementation of self attention
            result = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of self attention
            score = torch.matmul(Q, K.transpose(-2, -1)) * self.scale_factor  # / Q.size(-1) ** 0.5#self.context_length ** 0.5
            weight = score.masked_fill(self.low_tri[:, :, :T, :T] == 0, float('-inf'))
            soft_score = torch.softmax(weight, dim=-1)
            result = torch.matmul(soft_score, V)  # shape: (B, num_head T, head_size)
            result = self.drop(result)

        res = result.transpose(1, 2).reshape(B, T, C)
        res = self.linear(res)
        
        return res