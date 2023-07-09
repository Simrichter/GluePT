class params:
    def __init__(self, embedding_dimension, n_heads, n_blocks, batchsize, context_length, device, vocab_size, dropout):
        self.embedding_dimension = embedding_dimension
        self.num_heads = n_heads
        self.head_size = batchsize // n_heads
        self.n_blocks = n_blocks
        self.batchsize = batchsize
        self.context_length = context_length
        self.device = device
        self.vocab_size = vocab_size
        self.dropout = dropout
