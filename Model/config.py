class params:
    def __init__(self, embedding_dimension, n_heads, n_blocks, batchsize, context_length, device, vocab_size, dropout, task="prediction", bias=True, use_gpt2=False, freeze_model=False):
        self.embedding_dimension = embedding_dimension
        self.num_heads = n_heads
        self.head_size = batchsize // n_heads
        self.n_blocks = n_blocks
        self.batchsize = batchsize
        self.context_length = context_length
        self.device = device
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.task = task
        self.bias = bias
        self.use_gpt2 = use_gpt2
        self.freeze_model=freeze_model

        
class small_model (params):
    def __init__(self, batchsize, context_length, device, dropout, task="prediction", bias=True, use_gpt2=False, freeze_model=False):
        params.__init__(self, embedding_dimension=768, n_heads=12, n_blocks=12, batchsize=batchsize, context_length=context_length, device=device, vocab_size=50257, dropout=dropout, task=task, bias=bias, use_gpt2=use_gpt2, freeze_model=freeze_model)
        
class large_model (params):
    def __init__(self, batchsize, context_length, device, dropout, task="prediction", bias=True, use_gpt2=False, freeze_model=False):
        params.__init__(self, embedding_dimension=1024, n_heads=16, n_blocks=24, batchsize=batchsize, context_length=context_length, device=device, vocab_size=50257, dropout=dropout, task=task, bias=bias, use_gpt2=use_gpt2, freeze_model=freeze_model)