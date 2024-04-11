import torch
import torch.nn as nn
from torch.nn import functional as func
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config
import Model.Embeddings as Embeddings
import Model.config as config
import math
import os


class GptModel(nn.Module):

    def __init__(self, cfg, name):
        super().__init__()
        self.__use_gpt2 = cfg.use_gpt2
        self.freeze = cfg.freeze_model
        self.device = cfg.device
        self.T = cfg.context_length
        self.task = cfg.task
        self.name = name
        
        if self.__use_gpt2: # Load the complete GPT2 model instead of the own implementation. Only the output head is still created normally below.
            conf = GPT2Config(n_positions = cfg.context_length, n_embd = cfg.embedding_dimension, n_layer=cfg.n_blocks, n_head=cfg.num_heads, resid_pdrop=cfg.dropout, embd_pdrop=cfg.dropout, attn_pdrop=cfg.dropout) # Translates the config into a GPT2 config
            
            self.gpt2 = GPT2LMHeadModel(conf) if self.task == "pretraining" else GPT2LMHeadModel.from_pretrained(f"gpt2{'-medium' if cfg.embedding_dimension == 1024 else ''}")
            self.train()
            
        else:
            self.train()

            # Creates the token embedding matrix
            self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dimension)

            # If possible, loads the pretrained token embedding weights from gpt2 (small/medium). Note that huggingface calls the small model just "gpt2"
            if cfg.embedding_dimension in [768, 1024]:
                extension = '' if cfg.embedding_dimension==768 else '-medium' # No extension for loading GPT2-small, "-medium" extension for GPT2-medium
                path_to_weight = f"Model/token_embedding_weights{extension}.pt"
                if os.path.exists(path_to_weight):
                    tew = torch.load(path_to_weight)
                else:
                    print(f"Downloading token_embedding_weights from gpt2{extension}")
                    tew = GPT2Model.from_pretrained(f"gpt2{extension}").wte.weight
                    torch.save(tew, path_to_weight)

                # Overwrites the token embedding matrix with the loaded weights and freezes them (As they are already trained, we do not want to change them)
                self.token_embedding.weight = tew
                self.token_embedding.weight.requires_grad = False

            # Prepare a position embedding matrix capable of encoding cfg.context_length positions
            self.register_buffer('positional_embedding', Embeddings.get_positional_encoding(cfg))

            self.embedding_dropout = nn.Dropout(cfg.dropout)

            # The decoder stack consisting of multiple sequentially connected blocks
            self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_blocks)])

            # Since we use a pre-LN transformer (Layernorm in front of each block's sublayers instead of after), the output of the last decoder block has not been normalized. Therefore we apply this LayerNorm manually after the decoder stack
            self.norm = nn.LayerNorm(cfg.embedding_dimension)

        # Creates the output head depending on the task and the model setup
        if self.task == "pretraining":
            self.head = nn.Linear(cfg.embedding_dimension, cfg.vocab_size, bias=False)

            # If the model is pretrained, the weights of the token embedding layer and the linear layer in the output head are tyed.
            if self.__use_gpt2:
                self.head.weight = self.gpt2.transformer.wte.weight
            else:
                self.head.weight = self.token_embedding.weight
        else:
            out_dim = 1 if self.task == "stsb" else 3 if (self.task =="ax" or "mnli" in self.task) else 2
            self.ft_head = nn.Linear(cfg.embedding_dimension, out_dim) # Different Name is used for the head, so load_state_dict ignores the old language modelling head with (strict=False)
            self.ft_head.weight = torch.nn.Parameter(self.ft_head.weight/2)
            self.ft_head.bias = torch.nn.Parameter(self.ft_head.bias/2)
            

    def forward(self, x):
        if self.__use_gpt2:
            x = self.gpt2(x, output_hidden_states=True).hidden_states[-1] # If gpt2 is used, forward it and get the output of the decoder stack (The GPT2 output head is ignored)
        else:
            if self.freeze: # Freezing the model means that only the output head should be trained. Therefore everything else is calculated with disabled gradients and the input is forwarded in eval mode (important for dropouts)
                self.eval()
            with torch.set_grad_enabled(not self.freeze):
                x = self.token_embedding(x) # Token embedding
                if x.shape[1] <= self.positional_embedding.shape[1]:
                    x = x + self.positional_embedding[:, :x.shape[1], :] # Positional embedding
                else:
                    # If a sample is longer than the maximum context length, the fixed encoding formula is used to temporarily extend the encoding to the desired length
                    x = x + torch.cat((self.positional_embedding, torch.FloatTensor([Embeddings.get_single_encoding(self.positional_embedding.shape[2], i+self.positional_embedding.shape[1]) for i in range(x.shape[1]-self.positional_embedding.shape[1])])[None, :, :].to(self.device)), dim=1)
                x = self.embedding_dropout(x)

                # Forward the embedded input through the decoder stack (all blocks)
                x = self.norm(self.blocks(x))
            self.train() # In case, the model was frozen, train() ensures that the output head will still be trained
        
        # Pretraining and finetuning heads are named differently (to make loading a checkpoint easier).
        if self.task == "pretraining":
            return self.head(x)
        else:
            out = self.ft_head(x[:, -1:, :]) # When Finetuning only the last position in context_dim is forwarded to the head, because it holds full information
            return out
        

    # This method allows to sample text from the GPT model.
    # x is the input that will be continued by max_length new token.
    # If x is/grows larger than the maximum sequence length of the GPT, only the last T_{max} token are taken into account.
    def generate(self, x, max_length):
        if not self.hasattr("head") or self.head.weight.shape[1]!=50257 or self.task != "pretraining":
            print("Error: Output head is not suitable for text generation tasks")
            return
        for _ in range(max_length):
            logits = self(x.view(1, -1)[:, -self.T:]) # Appends a batch dimension of size 1 for compatibility and crops the input to a max length of T. Then implicitly calls the forward function
            last_logit = logits.view(-1, 50257)[-1,:] # Collapse the batch dimension and only use the last token position
            prob = func.softmax(last_logit, dim=-1) # Softmax is applied, since it is not done in the output head
            x_next = torch.multinomial(prob, num_samples=1) # Samples from the probability distribution. Alternatively, torch.argmax(probs) can be used
            x = torch.cat((x, x_next)) # Concatenates the sampled token to the input to be used in the next iteration
        return x

# This class defines a single decoder block in the decoder stack
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.sa = Multi_head_attention(cfg)
        self.norm1 = nn.LayerNorm(cfg.embedding_dimension)
        self.norm2 = nn.LayerNorm(cfg.embedding_dimension)
        # The feed-forward network is created as a torch.nn.Sequential layer
        self.ffwd = nn.Sequential(
            nn.Linear(cfg.embedding_dimension, 4 * cfg.embedding_dimension),
            nn.GELU(approximate='tanh'), # The tanh approximation of the GELU function is used to slightly speed up the calculations
            nn.Linear(4 * cfg.embedding_dimension, cfg.embedding_dimension),
            nn.Dropout(cfg.dropout)
        )
        # As described in "Language Models are Unsupervised Multitask Learners" all residual Layers (last layer before the residual path gets added)
        # are scaled by 1/sqrt(N) after initialization with stdv=0.02 (see "Improving Language Understanding by Generative Pre-Training")
        # since there are 2 residual connections per block, the number of residual layers N equals 2*cfg.n_blocks
        torch.nn.init.normal_(self.ffwd[2].weight, 0, 0.02 / math.sqrt(2 * cfg.n_blocks))
        # The second residual layer in the block is implemented inside the Multi_head_attention class below

    def forward(self, x):
        # Note that LayerNorm is applied before attention or feed-forward. Therefore this is a Pre-LN Transformer
        x = x + self.sa(self.norm1(x))
        x = x + self.ffwd(self.norm2(x))
        return x

# This class implements the masked multi-head self-attention, used in the decoder block
class Multi_head_attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        T, C = cfg.context_length, cfg.embedding_dimension
        nh, hs = cfg.num_heads, C // cfg.num_heads
        self.n_embd = C
        self.n_head = nh
        self.scale_factor = 1 / math.sqrt(hs) # The scaling factor used in the attention
        self.dropout = cfg.dropout

        # The efficient implementation uses only a single linear layer to create Q, K and V matrices for all heads
        if cfg.efficient_implementation:
            self.c_attn = nn.Linear(C, 3 * C, bias=cfg.bias)
        else:
            # The more intuitive, but slower implementation
            stdv = 1 / torch.sqrt(torch.tensor(C)) # Initializes the parameters with a uniform distribution (just like a torch.nn.linear layer)
            self.W_q = nn.Parameter(nn.init.uniform_(torch.empty(nh, C, hs), a=-stdv, b=stdv))
            self.W_k = nn.Parameter(nn.init.uniform_(torch.empty(nh, C, hs), a=-stdv, b=stdv))
            self.W_v = nn.Parameter(nn.init.uniform_(torch.empty(nh, C, hs), a=-stdv, b=stdv))
            if cfg.bias:
                self.B_q = nn.Parameter(nn.init.uniform_(torch.empty(hs)), a=-stdv, b=stdv)
                self.B_k = nn.Parameter(nn.init.uniform_(torch.empty(hs)), a=-stdv, b=stdv)
                self.B_v = nn.Parameter(nn.init.uniform_(torch.empty(hs)), a=-stdv, b=stdv)

        # The second residual layer, again scaled by 1/sqrt(N)
        self.linear = nn.Linear(C, C)
        torch.nn.init.normal_(self.linear.weight, 0, 0.02/math.sqrt(2*cfg.n_blocks))

        # Check if the own implementation or the torch implementation of the scaled dot product attention should be used
        # (both are equally fast, but the torch implementation prevents a bug that occurs in combination with automatic mixed precision)
        if not (torch.cuda.is_available() and hasattr(torch.nn.functional, 'scaled_dot_product_attention')):
            # The masking matrix is created
            self.register_buffer('mask', (torch.triu(torch.ones(T, T))*float('-inf')).view(1, 1, T, T))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape # Get the dimensions of the input for later use

        if hasattr(self, "W_q"):
            # Intuitive  but inefficient implementation is used
            x = x[:, None, :, :] # Introduce Dummy dimension for attention head dimension of the W_(q,k,v) weights, so that the batch dimension of x does not collide
            # Create Q, K and V
            Q = torch.matmul(x, self.W_q)
            K = torch.matmul(x, self.W_k)
            V = torch.matmul(x, self.W_v)
            if self.hasattr("B_q"):
                # Add bias values
                Q = Q+self.B_q
                K = K+self.B_k
                V = V+self.B_v
        else:
            # Efficient implementation with only a single linear transformation is used
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

            # Reorder the Q, K and V matrices to the desired shape (B, nh, T, hs)
            K = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            Q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            V = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if not hasattr(self, "mask"):
            # using the torch implementation of self attention
            result = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # using the manual implementation of self attention
            score = torch.matmul(Q, K.transpose(-2, -1)) * self.scale_factor
            weight = score+self.mask
            soft_score = torch.softmax(weight, dim=-1)
            result = torch.matmul(soft_score, V)
            result = self.drop(result)

        res = result.transpose(1, 2).reshape(B, T, C) # Concatenate the results of all heads
        res = self.linear(res) # Linear transformation after concatenation
        
        return res


# This function is used to create a PyTorch GPT model
# The standard setups for small and large model as well as the corresponding GPT2 sizes are configured as presets, otherwise default values are used.
# Loading weights from existing checkpoints and optionally compiling the model is also handled in this function
def create_model(name, epoch, device, task_name, dropout=0.0, compile_model=True, evaluate=False, embedding_dimension=768, num_heads=6, num_blocks=6, vocab_size=50257, context_dimension=256, bias=False, freeze_model=False, batch_size=256, **kwargs):
    if name=='small_model':
        parameters = config.small_model(batchsize=batch_size, context_length=context_dimension, device=device, dropout=dropout, task=task_name, freeze_model=freeze_model, **kwargs)
    elif name=='large_model':
        parameters = config.large_model(batchsize=batch_size, context_length=context_dimension, device=device, dropout=dropout, task=task_name, freeze_model=freeze_model, **kwargs)
    elif name=='gpt2_small':
        parameters = config.small_model(batchsize=batch_size, context_length=context_dimension, device=device, dropout=dropout, task=task_name, freeze_model=freeze_model, use_gpt2=True, **kwargs)
    elif name=='gpt2_medium':
        parameters = config.large_model(batchsize=batch_size, context_length=context_dimension, device=device, dropout=dropout, task=task_name, freeze_model=freeze_model, use_gpt2=True, **kwargs)
    else:                
        parameters = config.params(embedding_dimension=embedding_dimension, n_heads=num_heads, n_blocks=num_blocks, batchsize=batch_size, context_length=context_dimension, vocab_size=vocab_size, device=device, dropout=dropout, task=task_name, bias=bias, freeze_model=freeze_model, use_gpt2=False, **kwargs)
    
    model = GptModel(parameters, name).to(device)
    if torch.cuda.is_available() and compile_model and task_name=='pretraining':
        print('compiling model')
        model = torch.compile(model)
    
    # For mnli-m, mnli-mm and ax, the same finetuned model is used, which is saved with just the mnli prefix.
    file_task_name = 'mnli' if 'mnli' in task_name or task_name == 'ax' else task_name 
    
    path_to_checkpoint = f"Checkpoints/{name}/{name}.pt" if task_name=='pretraining' else (f"FinetunedModels/{name}/({epoch}){name}/{file_task_name}_({epoch}){name}.pt" if evaluate else f"Checkpoints/{name}/({epoch}){name}.pt" )

    # If possible, an existing checkpoint is loaded
    use_existing_model = os.path.exists(path_to_checkpoint) and not parameters.use_gpt2
    if use_existing_model:
        state = torch.load(path_to_checkpoint, map_location=device)
        if torch.cuda.is_available() and compile_model:
            sd = state["state_dict"]
        else:# This part is used if torch.compile is not available. Compiling a model adds a prefix to the names of the weights
            # remove '_orig_mod.' prefix to allow loading to an uncompiled Model
            sd = {k.removeprefix('_orig_mod.'): v for k, v in state["state_dict"].items()}

        model.load_state_dict(sd, strict=False)
        if task_name=='pretraining':
            print(f"Model {name} sucessfully loaded")
        else:
            print(f"Model ({epoch}){name} successfully loaded")
    elif not parameters.use_gpt2:
        assert task_name == 'pretraining', f"\n!!ERROR!!\nCould not find model at path {path_to_checkpoint}\nFinetuning requires an existing checkpoint"
        print("No model checkpoint loaded") # Only for pretraining it is desired to start from a fresh model
    return model, (state if "state" in locals() else None)
