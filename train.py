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
from torch.nn import functional as func
from datasets import load_dataset

import warnings

warnings.filterwarnings("ignore")

context_dimension = 256  # 1024  # Time
embedding_dimension = 768  # 192  # feature Channels (Should be multiple of num_heads)
accumulation_steps = 1
batch_size = 64 // accumulation_steps  # (micro)Batch 64
num_heads = 12  # for multi-head Attention
num_blocks = 12
vocab_size = 50257

model_name = '10set_owt_model'
use_existing_model = True #False#
use_karpathy = False  # True
compile_model = True

epochs = 10
num_workers = 4
train_test_percentage = 0.995

eval_interval = 500  # 250 # 1000
plot_intervall = 100
always_save_checkpoints = True
eval_tolerance = 5e-2

dropout = 0  # 0.35 #0.2
weight_decay = 1e-2
grad_clip = 5.0

learning_rate = 2.5e-4
min_lr = 1e-5
warmup_iterations = 400

best_loss = 1e8

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

# performance optimizations
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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

train_set, test_set = Data.Dataset(parameters, train=True, train_test_percentage=train_test_percentage), Data.Dataset(parameters, train=False, train_test_percentage=train_test_percentage)
train_sampler, test_sampler = Data.ResumableSampler(len(train_set)), Data.ResumableSampler(len(test_set))
train_loader, test_loader = DataLoader(train_set, batch_size=parameters.batchsize, num_workers=num_workers, pin_memory=True, sampler=train_sampler), DataLoader(test_set, batch_size=parameters.batchsize, num_workers=num_workers, pin_memory=True, sampler=test_sampler)#
#TODO Check if sampler can be changed after loading in DataLoader

model_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(model_params, learning_rate, (0.9, 0.95), weight_decay=weight_decay, foreach=False, fused=True)  # TODO Test if fused is good
scaler = GradScaler()

loss_history = {"train": [], "test": [], "test_interval": eval_interval, "plot_interval": plot_intervall}
start_iteration, start_epoch = 0, 0
if use_existing_model:
    state = torch.load('{}.pt'.format(model_name))
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scaler.load_state_dict(state["GradScaler"])
    train_sampler.set_state(state["samplers"]["train"])
    test_sampler.set_state(state["samplers"]["test"])
    start_iteration = state['iteration']
    start_epoch = state['epoch']
    loss_history = state['loss_history']

    print("Continuing from Model at iteration {} in epoch {}".format(start_iteration, start_epoch))


def get_lr(it):
    lr_decay_iters = len(train_loader) * epochs
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


def evaluate():
    model.eval()
    with torch.no_grad():
        losses = torch.zeros(len(test_loader))
        for k, batch in enumerate(tqdm(test_loader, leave=False)):
            x, y = batch[0].to(device), batch[1].to(device)  # get_batch(test=True)
            out = model(x)
            B, T, vs = out.shape
            loss = func.cross_entropy(out.view(B * T, vs), y.view(B * T), ignore_index=-1)
            losses[k] = loss.detach()  # Changed detach() and item()
    model.train()
    return losses.mean()


def save_state(iteration, epoch): #Checkpoint = True
    state_to_save = {
        "state_dict": model.state_dict(),
        "iteration": iteration + start_iteration,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "GradScaler": scaler.state_dict(),
        "samplers": {"train": train_sampler.get_state(), "test": test_sampler.get_state()},
        "loss_history": loss_history
    }
    torch.save(state_to_save, '{}.pt'.format(model_name))


def training_loop():  # iterations
    global best_loss

    # with trange(iteration_offset, iterations, initial=iteration_offset, total=iterations) as t:
    batch_loss = 0
    test_loss = 0 if len(loss_history['test'])==0 else loss_history['test'][-1] # TODO Check if this is better on gpu or cpu
    model.train()
    for epoch in range(start_epoch, epochs):
        for i, batch in enumerate(progressbar := tqdm(train_loader, initial=start_iteration, total = len(train_loader)+start_iteration)):
            step = i + epoch * len(train_loader)
            x, y = batch[0].to(device), batch[1].to(device)  # get_batch()
            with autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                out = model(x)
                B, T, vs = out.shape
                loss = func.cross_entropy(out.view(B * T, vs), y.view(B * T), ignore_index=-1) / accumulation_steps
            scaler.scale(loss).backward()
            batch_loss += loss.detach()
            del loss, x, y, out

            if ((step + 1) % accumulation_steps == 0) or (step + 1 == len(train_loader)):
                if step % plot_intervall == 0:
                    loss_history['train'].append(batch_loss.item())
                    progressbar.set_postfix({'train_loss': batch_loss.item(), 'test_loss': test_loss})  # Changed Indentation
                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                for g in optimizer.param_groups:
                    g['lr'] = get_lr(step + epoch * train_loader.__len__())
                batch_loss = 0

            if (step + 1) % eval_interval == 0:
                test_loss = evaluate().detach().item()
                loss_history['test'].append(test_loss)
                # progressbar.set_postfix({'train_loss': loss.item(), 'test_loss': test_loss.item()})
                if always_save_checkpoints or test_loss < best_loss + eval_tolerance:
                    best_loss = test_loss
                    save_state(i, epoch)
                else:
                    print("test loss got larger, no checkpoint will be saved")


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
