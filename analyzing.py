import numpy as np
import torch
import Model.Gpt as gpt
import Model.config as config
import Model.Embeddings as emb
import matplotlib.pyplot as plt
import math

context_dimension = 512#256  # Time
embedding_dimension = 768#384  # feature Channels (Should be multiple of num_heads)
batch_size = 16  # Batch
num_heads = 12  # for multi-head Attention
num_blocks = 12
vocab_size = 50257  # TODO Use 50304 (next multiple of 64) for efficiency ??
dropout = 0.3

use_existing_model = False
#model_name = 'shakespeare.pt'
model_name = 'huge_bc_model.pt'
training_iterations = 20000  # 6000
iteration_offset = 0
eval_iters = 50
eval_interval = 500
always_save_checkpoints = True
eval_tolerance = 5e-2
train_test_percentage = 0.8
learning_rate = 1e-3  # 5e-4
min_lr = 1e-4
warmup_iterations = 100
lr_decay_iters = training_iterations

best_loss = 1e8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#data = torch.load('shakespeare_tokenized.pt').to(device)
#data = torch.load('bc1.pt', map_location=torch.device('cpu')).to(device)
#print('{0:,}'.format(len(data)))

#train_split = data[:int(train_test_percentage * len(data))]
#test_split = data[int(train_test_percentage * len(data)):]

parameters = config.params(embedding_dimension, n_heads=num_heads, n_blocks=num_blocks,
                           batchsize=batch_size, context_length=context_dimension, vocab_size=vocab_size,
                           device=device, dropout=dropout)

#model = gpt.GptModel(parameters).to(device)
#if use_existing_model:
#state = torch.load('final_huge_bc_model.pt', map_location=torch.device('cpu'))
#model.load_state_dict(state['state_dict'])
#optimizer.load_state_dict(state['optimizer'])
#iteration_offset = state['iteration']
#loss_history = state['loss_history']

#trai = [i.item() for i in loss_history['train']]
#tes = [i.item() for i in loss_history['test']]
#state['loss_history']['train'] = trai
#state['loss_history']['test'] = tes
#torch.save(state, 'reduced_save.pt')


def get_batch(test=False):
    if test:
        d = 1#test_split
    else:
        d = 1#train_split
    # -2 because len(d) is 1 larger than the last index of d and y needs a shift to the right by 1
    indizes = torch.randint(low=0, high=max(len(d) - context_dimension, 0), size=(batch_size,))
    x = torch.stack([d[i:i + context_dimension] for i in indizes]).to(device)
    y = torch.stack([d[i + 1:i + context_dimension + 1] for i in indizes]).to(device)

    return x, y

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iterations:
        return learning_rate * it / warmup_iterations
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iterations) / (lr_decay_iters - warmup_iterations)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

x = range(training_iterations)
y = [get_lr(i) for i in x]
plt.plot(x, y)
plt.show()

#x = get_batch()[0]
#print(x[0].shape)
#print(x[0].tolist()[0])
#print(emb.decode(x[0].tolist()))

s1 = "First Citizen:\nBefore we proceed any further, hear me speak."
s3 = "First Citizen: Before we proceed any further, hear me speak."
s2 = "First Citizen \nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?"
#model.eval()
#print(emb.decode(
 #   model.generate(torch.tensor(get_batch()[0]).to(device), 100).tolist()))
#model.train()
#x = emb.encode(s3)
#x = torch.tensor(x).view(1, -1)
#print(x.shape)
#logits, loss = model(x)#[:, -context_dimension:])
#logits = logits[:, -1, :]

#v, _ = torch.topk(logits, 10)
#logits[logits < v[:, [-1]]] = -float('Inf')

#logits = F.softmax(logits, dim=-1)

#print(emb.decode([torch.argmax(logits, dim=-1).item()]))
 #           logits = logits[:, -1, :]