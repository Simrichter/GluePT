import torch
import math
import Model.Gpt as gpt
import Model.config as config
import Model.Embeddings as emb
from tqdm import tqdm, trange

context_dimension = 512  # Time
embedding_dimension = 768  # 192#384  # feature Channels (Should be multiple of num_heads)
batch_size = 16 #64  # Batch
num_heads = 12  # for multi-head Attention
num_blocks = 12
vocab_size = 50257
dropout = 0.2

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
lr_decay_iters = 50000 #training_iterations

best_loss = 1e8

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')

data = torch.load('Bookcorpus/bc1.pt').to(device)
#data = torch.load('shakespeare_tokenized.pt').to(device)
#print(data.shape)

train_split = data[:int(train_test_percentage * len(data))]
test_split = data[int(train_test_percentage * len(data)):]
#print("data loaded")

parameters = config.params(embedding_dimension, n_heads=num_heads, n_blocks=num_blocks,
                           batchsize=batch_size, context_length=context_dimension, vocab_size=vocab_size,
                           device=device, dropout=dropout)

model = torch.compile(gpt.GptModel(parameters).to(device))
# model = gpt.GptModel(parameters).to(device)
model_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(model_params, learning_rate, foreach=False, fused=True)  # TODO Test if fused is good
loss_history = {"train": [], "test": [], "test_interval": eval_interval}

if use_existing_model:
    state = torch.load('final_{}'.format(model_name))
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    iteration_offset = state['iteration']
    loss_history = state['loss_history']
    
    print("Continuing from Model at iteration", iteration_offset, ", best test loss:", loss_history['test'][-1])


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


def get_batch(test=False):
    if test:
        d = test_split
    else:
        d = train_split
    # -2 because len(d) is 1 larger than the last index of d and y needs a shift to the right by 1
    indizes = torch.randint(low=0, high=max(len(d) - context_dimension, 0), size=(batch_size,))
    x = torch.stack([d[i:i + context_dimension] for i in indizes]).to(device)
    y = torch.stack([d[i + 1:i + context_dimension + 1] for i in indizes]).to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(test=True)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    test_loss = losses.mean()
    model.train()
    return test_loss


def save_state(iteration, checkpoint=True):
    state_to_save = {
        "state_dict": model.state_dict(),
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        "loss_history": loss_history
    }
    if checkpoint:
        torch.save(state_to_save, model_name)
    else:
        torch.save(state_to_save, 'final_{}'.format(model_name))  # Used to save a model that may be overfitted


def training_loop(iterations):
    global best_loss

    #ascii=" ▖▘▝▗▚▞█", ascii="░▒█", ascii=' >='
    with trange(iteration_offset, iterations, initial=iteration_offset, total=iterations) as t:
        for steps in t:
            optimizer.zero_grad(set_to_none=True)
            x, y = get_batch()
            _, loss = model(x, y)
            if (steps)%10==0:
                loss_history['train'].append(loss.item())
            if (steps+1) % eval_interval == 0:
                test_loss = estimate_loss()
                loss_history['test'].append(test_loss.item())
                # print("Iteration ", steps, " train loss: ", loss.item(), " test loss: ", test_loss.item())
                t.set_postfix({'train_loss': loss.item(), 'test_loss': test_loss.item()})
                if test_loss < best_loss + eval_tolerance or always_save_checkpoints:
                    best_loss = test_loss
                    save_state(steps)
                #else:
                    #print("test loss got larger, no checkpoint will be saved")

            loss.backward()
            optimizer.step()
            for g in optimizer.param_groups:
                g['lr'] = get_lr(steps)


def param_count(module=model):
    model_parameters = filter(lambda p: p.requires_grad, module.parameters())
    params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    return params


start_index = 0

print("Number of Parameters: ", '{0:,}'.format(param_count()))
training_loop(training_iterations)
save_state(training_iterations, checkpoint=False)
# print(emb.decode(model.generate(torch.tensor(emb.encode(s2)).view(1,-1).to(device), 100)[0].tolist()))
model.eval()
# print(emb.decode(model.generate(train_split[start_index:context_dimension-127+start_index], 100).tolist()))
# print(emb.decode(model.generate(train_split[0:1], 100).tolist()))
print(emb.decode(model.generate(torch.tensor(emb.encode(' '), device=device), 150).tolist()))
model.train()
