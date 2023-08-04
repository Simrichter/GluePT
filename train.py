import torch
import math
import Data
import Model.Gpt as gpt
import Model.Karpathy_model as Karpathy_model
import Model.config as config
import Model.Embeddings as emb
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch import autocast

import warnings

warnings.filterwarnings("ignore")

context_dimension = 256  # 1024  # Time
embedding_dimension = 768  # 192  # feature Channels (Should be multiple of num_heads)
accumulation_steps = 1  # 2
batch_size = 32 // accumulation_steps  # (micro)Batch
num_heads = 12  # for multi-head Attention
num_blocks = 12
vocab_size = 50257

model_name = 'mixed_bc_model'
use_existing_model = False
use_karpathy = False  # True
compile_model = True
# model_name = 'reg_shakespeare'

# training_iterations = 5000  # 6000
iteration_offset = 0
epochs = 1
num_workers = 4
train_test_percentage = 0.99

eval_interval = 2000  # 250 # 1000
always_save_checkpoints = True
eval_tolerance = 5e-2

dropout = 0  # 0.35 #0.2
weight_decay = 0  # 1e-1
grad_clip = 0.0

learning_rate = 7e-4
min_lr = 8e-5
warmup_iterations = 100
# lr_decay_iters = 50000  # training_iterations

best_loss = 1e8

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

# performance optimizations
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# data = torch.load('Bookcorpus/bc1.pt').to(device)
# data = torch.load('shakespeare_tokenized.pt').to(device)
# print(data.shape)

# train_split = data[:int(train_test_percentage * len(data))]
# test_split = data[int(train_test_percentage * len(data)):]
# print("data loaded")
parameters = config.params(embedding_dimension, n_heads=num_heads, n_blocks=num_blocks,
                           batchsize=batch_size, context_length=context_dimension, vocab_size=vocab_size,
                           device=device, dropout=dropout)
if use_karpathy:
    k_parameters = Karpathy_model.GPTConfig(block_size=context_dimension, vocab_size=vocab_size, n_layer=num_blocks,
                                            n_head=num_heads, n_embd=embedding_dimension, dropout=dropout, bias=False)
    model = Karpathy_model.GPT(k_parameters).to(device)
else:
    model = gpt.GptModel(parameters).to(device)
if torch.cuda.is_available() and compile_model:
    print('compiling model')
    model = torch.compile(model)

train_set = Data.Dataset(parameters, train=True, train_test_percentage=train_test_percentage)
test_set = Data.Dataset(parameters, train=False, train_test_percentage=train_test_percentage)
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)

model_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(model_params, learning_rate, weight_decay=weight_decay, foreach=False,
                              fused=True)  # TODO Test if fused is good
scaler = GradScaler()
loss_history = {"train": [], "test": [], "test_interval": eval_interval}

if use_existing_model:
    state = torch.load('final_{}.pt'.format(model_name))
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    iteration_offset = state['iteration']
    loss_history = state['loss_history']

    print("Continuing from Model at iteration", iteration_offset, ", best test loss:", loss_history['test'][-1])


def get_lr(it):
    lr_decay_iters = train_loader.__len__() * epochs
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


'''def get_batch(test=False):
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
    #TODO torch.set_grad_enabled(False)
    #with torch.no_grad():
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(test=True)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    test_loss = losses.mean()
    model.train()
    return test_loss'''


def evaluate():
    model.eval()
    with torch.no_grad():
        losses = torch.zeros(test_loader.__len__())
        for k, batch in enumerate(test_loader):
            X, Y = batch[0].to(device), batch[1].to(device)  # get_batch(test=True)
            logits, loss = model(X, Y)
            losses[k] = loss.detach()  # Changed detach() and item()
    model.train()
    return losses.mean()


def save_state(iteration, checkpoint=True):
    state_to_save = {
        "state_dict": model.state_dict(),
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        "loss_history": loss_history
    }
    if checkpoint:
        torch.save(state_to_save, '{}.pt'.format(model_name))
    else:
        torch.save(state_to_save, 'final_{}.pt'.format(model_name))  # Used to save a model that may be overfitted


def training_loop():  # iterations
    global best_loss

    # with trange(iteration_offset, iterations, initial=iteration_offset, total=iterations) as t:
    batch_loss = 0
    test_loss = torch.tensor(0)  # TODO Check if this is better on gpu or cpu
    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(progressbar := tqdm(train_loader)):
            x, y = batch[0].to(device), batch[1].to(device)  # get_batch()
            with autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                _, loss = model(x, y)
                del x
                del y
                loss = loss / accumulation_steps
                batch_loss += loss.detach()
            scaler.scale(loss).backward()
            del loss

            if ((step + 1) % accumulation_steps == 0) or (step + 1 == len(train_loader)):
                if (step) % 10 == 0:
                    loss_history['train'].append(batch_loss.item())
                progressbar.set_postfix({'train_loss': batch_loss.item(), 'test_loss': test_loss.item()})
                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0

            if (step + epoch * train_loader.__len__() + 1) % eval_interval == 0:
                test_loss = evaluate()  # estimate_loss()
                loss_history['test'].append(test_loss.detach().item())
                # print("Iteration ", steps, " train loss: ", loss.item(), " test loss: ", test_loss.item())
                # progressbar.set_postfix({'train_loss': loss.item(), 'test_loss': test_loss.item()})
                if test_loss < best_loss + eval_tolerance or always_save_checkpoints:
                    best_loss = test_loss
                    save_state(step + epoch * train_loader.__len__())
                else:
                    print("test loss got larger, no checkpoint will be saved")

            for g in optimizer.param_groups:
                g['lr'] = get_lr(step + epoch * train_loader.__len__())


def param_count(module=model):
    model_parameters = filter(lambda p: p.requires_grad, module.parameters())
    params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    return params


start_index = 0

print("Number of Parameters: ", '{0:,}'.format(param_count()))
training_loop()  # training_iterations
save_state(train_loader.__len__() * epochs, checkpoint=False)
# print(emb.decode(model.generate(torch.tensor(emb.encode(s2)).view(1,-1).to(device), 100)[0].tolist()))
model.eval()
# print(emb.decode(model.generate(train_split[start_index:context_dimension-127+start_index], 100).tolist()))
# print(emb.decode(model.generate(train_split[0:1], 100).tolist()))
print(emb.decode(model.generate(torch.tensor(emb.encode(' '), device=device), 150).tolist()))
model.train()
