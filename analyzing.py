import torch
from torch.nn import functional as F
import Model.Gpt as gpt
import Model.config as config
import Model.Embeddings as emb

context_dimension = 128#256  # Time
embedding_dimension = 192#384  # feature Channels (Should be multiple of num_heads)
batch_size = 32  # Batch
num_heads = 6  # for multi-head Attention
num_blocks = 6
vocab_size = 50257  # TODO Use 50304 (next multiple of 64) for efficiency ??

use_existing_model = True
training_iterations = 2000
eval_iters = 50
eval_interval = 200
train_test_percentage = 0.9
learning_rate = 5e-4
dropout = 0.2  # TODO implement dropout

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = torch.load('shakespeare_tokenized.pt').to(device)
train_split = data[:int(train_test_percentage * len(data))]
test_split = data[int(train_test_percentage * len(data)):]

parameters = config.params(embedding_dimension, n_heads=num_heads, n_blocks=num_blocks,
                           batchsize=batch_size, context_length=context_dimension, vocab_size=vocab_size,
                           device=device)

model = gpt.GptModel(parameters).to(device)

model.load_state_dict(torch.load('Overfitt_model_save.pt', map_location=device))

s1 = "First Citizen:\nBefore we proceed any further, hear me speak."
s3 = "First Citizen: Before we proceed any further, hear me speak."
s2 = "First Citizen \nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?"
model.eval()
print(emb.decode(
    model.generate(torch.tensor(emb.encode(s3)).to(device), 100).tolist()))
#model.train()
x = emb.encode(s3)
x = torch.tensor(x).view(1, -1)
print(x.shape)
logits, loss = model(x)#[:, -context_dimension:])
logits = logits[:, -1, :]

#v, _ = torch.topk(logits, 10)
#logits[logits < v[:, [-1]]] = -float('Inf')

#logits = F.softmax(logits, dim=-1)

print(emb.decode([torch.argmax(logits, dim=-1).item()]))
 #           logits = logits[:, -1, :]