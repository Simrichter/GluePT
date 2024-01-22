import torch
import math
import Data
import Model.Gpt as gpt
import Model.config as config
import Model.Embeddings as emb
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch import autocast
from torch.nn import functional as func
import os
import datetime
from torch.multiprocessing import set_start_method
import warnings

# warnings.filterwarnings("ignore")

context_dimension = 256  # 1024  # Time
embedding_dimension = 768 #1024 # feature Channels (Should be multiple of num_heads)
batch_size = 256
accumulation_steps = 4
assert batch_size % accumulation_steps == 0  # batch_size must be multiple of accumulation_steps
micro_batch_size = batch_size // accumulation_steps # should result in ~64 to fit the model on a Nvidia A100 GPU (in current configuration)
num_heads = 12 # 16 # for multi-head Attention
num_blocks = 12 #24 #
vocab_size = 50257

model_name = 'small_model'#'NEWset_owt_model'#'large_model'
use_existing_model = os.path.exists(f'Checkpoints/{model_name}/{model_name}.pt') #
compile_model = True#False#
stop_time = datetime.datetime(year=2023, month=11, day=28, hour=7, minute=59)
gpu_num = 3

epochs = 2
target_epochs = 15 # this is used for the learning rate schedule, if only a few epochs are trainined, but the learning rate should be comparable to a full training, this should be set to the amount of epochs in the full training and the 'epochs' parameter is set to the actual epochs
num_workers = 2
train_test_percentage = 0.995

eval_interval = 3448#1000
plot_interval = 100
always_save_checkpoints = True
eval_tolerance = 5e-2

dropout = 0  # 0.35 #0.2
weight_decay = 1e-2
grad_clip = 1.0

learning_rate = 2.5e-4
min_lr = 1e-5
warmup_iterations = 400

device = torch.device(f'cuda:{gpu_num}') if torch.cuda.is_available() else 'cpu' #TODO

# performance optimizations
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

parameters = config.params(embedding_dimension, n_heads=num_heads, n_blocks=num_blocks,
                           batchsize=batch_size, context_length=context_dimension, vocab_size=vocab_size,
                           device=device, dropout=dropout)
model = gpt.GptModel(parameters).to(device)
if torch.cuda.is_available() and compile_model:
    print('compiling model')
    model = torch.compile(model)

# Loading the Datasets
train_set = Data.Dataset(parameters, train=True, train_test_percentage=train_test_percentage)
test_set = Data.Dataset(parameters, train=False, train_test_percentage=train_test_percentage)
# Creating custom samplers to make resuming inside an epoch possible
train_sampler = Data.ResumableSampler(len(train_set))
test_sampler = Data.ResumableSampler(len(test_set))

# Two parameter groups, because weight decay is not applied to bias or gain weights
decay_groups = [{'params': [p for p in filter(lambda p: p.requires_grad and p.dim() >= 2, model.parameters())], 'weight_decay': weight_decay},
                {'params': [p for p in filter(lambda p: p.requires_grad and p.dim() < 2, model.parameters())], 'weight_decay': 0.0}]
optimizer = torch.optim.AdamW(decay_groups, 0, (0.9, 0.95), fused=torch.cuda.is_available()) # Use the fused implementation when on a GPU
scaler = GradScaler()

loss_history = {"running_loss": [], "train": [], "test": [], "test_interval": eval_interval,
                "plot_interval": plot_interval}

start_iteration, start_epoch = 0, 0

if use_existing_model:
    state = torch.load(f'Checkpoints/{model_name}/{model_name}.pt', map_location = device)
    start_iteration = state['iteration']
    start_epoch = state['epoch']
    
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scaler.load_state_dict(state["GradScaler"])
    train_sampler.set_state(state["samplers"]["train"], (start_iteration)*batch_size)
    test_sampler.set_state(state["samplers"]["test"], -1)
    
    loss_history = state['loss_history']
    
    num_workers = 0 # Needed to prevent a bug that occurs when continuing training with multiple dataloader worker threads

    print("Continuing at iteration {} in epoch {}".format(start_iteration+1, start_epoch))

# Creating the DataLoaders (test_loader does not use Gradient Accumulation, thus micro_batch_size is used)
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                          sampler=train_sampler, )
test_loader = DataLoader(test_set, batch_size=micro_batch_size, num_workers=num_workers, pin_memory=True,
                         sampler=test_sampler)
    
def get_lr(it):
    total_iters = len(train_loader) * target_epochs
    # warmup
    if it < warmup_iterations:
        return learning_rate * it / warmup_iterations
    # added warmup compatibility to formula from https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    if warmup_iterations <= it <= total_iters:
        return min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos((it - warmup_iterations) * math.pi / (total_iters - warmup_iterations)))
    # when training further than initially planned, stay at min_lr
    if it > total_iters:
        return min_lr


def evaluate():
    eval_iters = 694 #Used to limit the evaluation on the training set to roughly the same scope as the evaluation on test
    model.eval()
    with torch.no_grad():
        losses = {'test': torch.zeros(len(test_loader)), 'train': torch.zeros(eval_iters+1)}
        for k, batch in enumerate(tqdm(test_loader, leave=False, desc='evaluating Test', colour='yellow')):
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            B, T, vs = out.shape
            loss = func.cross_entropy(out.view(B * T, vs), y.view(B * T), ignore_index=-1)  # ignore last index
            losses['test'][k] = loss.detach()
        eval_train_loader = DataLoader(train_set, batch_size=micro_batch_size, num_workers=num_workers, pin_memory=True,shuffle=True)
        for k, batch in enumerate(tqdm(eval_train_loader, total=eval_iters, leave=False, desc='evaluating Train', colour='yellow')):
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            B, T, vs = out.shape
            loss = func.cross_entropy(out.view(B * T, vs), y.view(B * T), ignore_index=-1)  # ignore last index
            losses['train'][k] = loss.detach()
            if k >= eval_iters:
                break
    model.train()
    return losses['train'].mean().item(), losses['test'].mean().item()


def save_state(iteration, epoch, name_prefix = ""):
    state_to_save = {
        "state_dict": model.state_dict(),
        "iteration": iteration + start_iteration,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "GradScaler": scaler.state_dict(),
        "samplers": {"train": train_sampler.get_state(), "test": test_sampler.get_state()},
        "loss_history": loss_history
    }
    path = os.path.join('Checkpoints', f'{model_name}')
    if not os.path.exists(path):
        os.makedirs(path)
    if name_prefix == "" and epoch <= 2: #Save 10% epochs only for the first 2 epochs.
        name_prefix = f"{epoch},{-(-iteration//3448)}_" #ceil division to remove long decimal places from the file name (the two "-" and the floor division create a ceil division)
    torch.save(state_to_save, f'{path}/{name_prefix}{model_name}.pt')


def training_loop():
    global start_iteration, start_epoch, train_sampler
    
    begin = 0 if start_iteration <= 0 else start_iteration + 1

    batch_loss = torch.tensor(0.0, device=device)
    train_loss = loss_history['train'][-1] if len(loss_history['train']) > 0 else 0
    test_loss = loss_history['test'][-1] if len(loss_history['test']) > 0 else 0  # TODO Check if this is better on gpu or cpu
    model.train()
    for epoch in range(start_epoch, epochs):
        for i, multi_batch in enumerate(progressbar := tqdm(train_loader, desc=f'epoch {epoch}', initial = begin, total=len(train_loader) + start_iteration, colour='cyan')):
            
            # Stops training if the end of a reserved GPU timeslot is reached.
            if isinstance(stop_time, datetime.datetime) and datetime.datetime.now() >= stop_time: #TODO versuch, die datetime deaktivierbar zu machen
                break
            
            step = i + begin + epoch * (len(train_loader)+start_iteration)  # calculating the global step so the correct evaluation interval remains over epochs
            for x, y in zip(multi_batch[0].split(micro_batch_size), multi_batch[1].split(micro_batch_size)):
                x, y = x.to(device), y.to(device)
                with autocast(device_type='cuda', dtype=torch.float16, enabled=True):  # using autocast and Automatic Mixed Precision for training speedup
                    out = model(x)
                    B, T, vs = out.shape
                    loss = func.cross_entropy(out.view(B * T, vs), y.view(B * T), ignore_index=-1) / accumulation_steps
                scaler.scale(loss).backward()
                batch_loss += loss.detach()
                del loss, x, y, out

            if step % plot_interval == 0:
                loss_history['running_loss'].append(batch_loss.item())
                progressbar.set_postfix({'running_loss': batch_loss.item(), 'train_loss': train_loss,'test_loss': test_loss})
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            for g in optimizer.param_groups:
                g['lr'] = get_lr(step)
            

            if step % eval_interval == 0:
                train_loss, test_loss = evaluate()
                loss_history['test'].append(test_loss)
                loss_history['train'].append(train_loss)
                progressbar.set_postfix({'running_loss': batch_loss.item(), 'train_loss': train_loss,'test_loss': test_loss})
                save_state(i, epoch)
            batch_loss = torch.tensor(0.0, device=device) # Zeroing the batch_loss for the next iteration
        else:
            # If no break occurs and the inner loop finishes execution normally, this gets executed at the end of each epoch
            start_iteration = 0
            begin = 0
            save_state(0, epoch, name_prefix=f'({epoch+1})')
            continue # Skips the following line
        break # This is used to propagate a break in the inner loop to the outer loop


def param_count(module=model):
    model_parameters = filter(lambda p: p.requires_grad, module.parameters())
    params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    return params


start_index = 0

print("Number of Parameters: ", '{0:,}'.format(param_count()))
training_loop()  # training_iterations
#save_state(train_loader.__len__(), epochs)
# print(emb.decode(model.generate(torch.tensor(emb.encode(s2)).view(1,-1).to(device), 100)[0].tolist()))
#model.eval()
# print(emb.decode(model.generate(train_split[start_index:context_dimension-127+start_index], 100).tolist()))
# print(emb.decode(model.generate(train_split[0:1], 100).tolist()))
#print(emb.decode(model.generate(torch.tensor(emb.encode(' '), device=device), 150).tolist()))
#model.train()
