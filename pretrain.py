import torch
import math
import Data
import Model.Gpt as Gpt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch import autocast
from torch.nn import functional as func
import os
import datetime

def get_lr(it, total_iters, max_lr, min_lr):
        warmup_iterations = 4000 # This number is drawn from "Attention is all you need" Paper
        # warmup
        if it < warmup_iterations:
            return max_lr * it / warmup_iterations
        # Similar to the formula from https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html, but with added warmup compatibility
        if warmup_iterations <= it <= total_iters:
            return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos((it - warmup_iterations) * math.pi / (total_iters - warmup_iterations)))
        # when training further than initially planned, stay at min_lr
        if it > total_iters:
            return min_lr

def save_state(iteration, model, optimizer, scaler, train_sampler, test_sampler, epoch, epoch_length, loss_history, detailed, name_prefix = ""):
        state_to_save = {
            "state_dict": model.state_dict(),
            "iteration": iteration,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "GradScaler": scaler.state_dict(),
            "samplers": {"train": train_sampler.get_state(), "test": test_sampler.get_state()},
            "loss_history": loss_history
        }
        path = os.path.join('Checkpoints', f'{model.name}')
        if not os.path.exists(path):
            os.makedirs(path)
        
        # If detailed mode is active, save 10% epochs for the first 2 epochs.
        # The first time, a checkpoint reaches a new 10%, it is saved.
        # This guarantees that a checkpoint is not overwritten by later saves that have more training time but are rounded to the same "name_prefix"
        # For example, a checkpoint at 1.11 epochs cannot be overwritten by a checkpoint at 1.19 epochs
        if name_prefix == "":
            # If no prefix is given, check if detailed mode should automatically create a checkpoint
            if detailed and  epoch < 2:
                epoch_float = epoch + (iteration // (epoch_length//10))/10
                if isinstance(epoch_float, float) and epoch_float.is_integer():
                    epoch_float = int(epoch_float) # If the prefix would be 1.0, 2.0, etc. only the integer value is used (To follow the naming scheme when detailed mode is not active)
                potential_prefix = f"({epoch_float})"
                if not os.path.exists(f"{path}/{potential_prefix}{model.name}.pt"):
                    torch.save(state_to_save, f"{path}/{potential_prefix}{model.name}.pt")
        else:
            # If a name_prefix is given, save a checkpoint with that prefix
            torch.save(state_to_save, f"{path}/{name_prefix}{model.name}.pt")

        # Always save a checkpoint without name prefix, which represents the most recent state. It is used to resume training.
        torch.save(state_to_save, f"{path}/{model.name}.pt")

def evaluate(model, device, test_loader, train_set, micro_batch_size, num_workers):
    eval_iters=len(test_loader)
    model.eval()
    with torch.no_grad():
        # print("Length of the test_loader is:", len(test_loader))
        losses = {'test': torch.zeros(eval_iters), 'train': torch.zeros(eval_iters+1)}

        # Evaluates on the validation set
        for k, batch in enumerate(tqdm(test_loader, leave=False, desc='evaluating Test', colour='yellow')):
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            B, T, vs = out.shape
            loss = func.cross_entropy(out.view(B * T, vs), y.view(B * T), ignore_index=-1)  # ignore last index in the calculation of the loss, as it does not have a label
            losses['test'][k] = loss.detach()

        # Evaluates on random data from the train set (Gives a loss curve that is not affected by dropout and other training artifacts)
        eval_train_loader = DataLoader(train_set, batch_size=micro_batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), shuffle=True)
        for k, batch in enumerate(tqdm(eval_train_loader, total=eval_iters, leave=False, desc='evaluating Train', colour='yellow')):
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            B, T, vs = out.shape
            loss = func.cross_entropy(out.view(B * T, vs), y.view(B * T), ignore_index=-1)  # ignore last index
            losses['train'][k] = loss.detach() 
            if k >= eval_iters: # Used to limit the evaluation on the training set to roughly the same scope as the evaluation on test
                break
    model.train()
    return losses['train'].mean().item(), losses['test'].mean().item()

def train(model, state, epochs, device, num_subsets, detailed = False, batch_size=256, accumulation_steps=4, max_lr = 2.5e-4, min_lr=1e-5, n_workers=3, evals_per_epoch=10, plot_interval = 100, stop_time=None, weight_decay=1e-2, grad_clip=1.0):

    if detailed and evals_per_epoch<10:
        print("At least 10 evaluations per epoch are necessary for detailed mode, setting has been updated accordingly")
        evals_per_epoch = 10

    # Deactivates dataloading with multiple workers, when continuing from a checkpoint (This is needed to prevent a bug)
    num_workers = n_workers if state==None else 0

    assert batch_size%accumulation_steps==0, "Batch size must be divisible by the number of accumulation steps"
    micro_batch_size=batch_size//accumulation_steps

    # performance optimizations
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Loading the Datasets
    train_set = Data.Dataset(model.T, True, num_subsets)
    test_set = Data.Dataset(model.T, False, num_subsets)

    # Creating custom samplers to make resuming inside an epoch possible
    train_sampler = Data.ResumableSampler(len(train_set))
    test_sampler = Data.ResumableSampler(len(test_set))

    # Creating the DataLoaders (test_loader does not use Gradient Accumulation. Micro_batch_size is used, since we know that it will fit on the GPU)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), sampler=train_sampler)
    test_loader = DataLoader(test_set, batch_size=micro_batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), sampler=test_sampler)
    
    # Setting the interval, which defines the number of batches between two evaluations
    eval_interval = len(train_loader)//evals_per_epoch

    # Parameters are placed in two groups, because weight decay is not applied to bias or gain weights
    decay_groups = [{'params': [p for p in filter(lambda p: p.requires_grad and p.dim() >= 2, model.parameters())], 'weight_decay': weight_decay},
                    {'params': [p for p in filter(lambda p: p.requires_grad and p.dim() < 2, model.parameters())], 'weight_decay': 0.0}]
    
    # Instantiates the optimizer and uses the fused implementation when on a GPU since it is faster
    optimizer = torch.optim.AdamW(decay_groups, 0, (0.9, 0.95), fused=torch.cuda.is_available())

    # Instantiates the gradient scaler (but it is only active when cuda is available)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # A dictionary storing all values throughout training. Used for plotting
    loss_history = {"running_loss": [], "train": [], "test": [], "test_interval": eval_interval, "plot_interval": plot_interval}

    # These variables control where to start pretraining (necessary for continuing from a checkpoint)
    start_iteration, start_epoch = 0, 0

    # If a checkpoint is loaded, set all necessary values and states
    if state is not None:
        start_iteration = state['iteration']+1
        start_epoch = state['epoch']
        
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state["GradScaler"])
        train_sampler.set_state(state["samplers"]["train"], (start_iteration)*batch_size)
        test_sampler.set_state(state["samplers"]["test"], -1)
        
        loss_history = state['loss_history']

        print(f"Continuing at iteration {start_iteration} in epoch {start_epoch}")

    # begin = 0 if start_iteration <= 0 else start_iteration + 1 # An offset when starting from a checkpoint

    # Variables used for displaying in the progress bar
    train_loss = loss_history['train'][-1] if len(loss_history['train']) > 0 else 0
    test_loss = loss_history['test'][-1] if len(loss_history['test']) > 0 else 0
    batch_loss = torch.tensor(0.0, device=device)
    
    # Sets the model in train mode (important for dropout layers)
    model.train()

    # Performs an initial evaluation of the model before any training takes place
    # Also saves the fully untrained model as the checkpoint for epoch 0
    # If pretraining is resumed from a checkpoint, this initial evaluation is skipped 
    if state is None:
        train_loss, test_loss = evaluate(model, device, test_loader, train_set, micro_batch_size, num_workers)
        loss_history['test'].append(test_loss)
        loss_history['train'].append(train_loss)
        save_state(0, model, optimizer, scaler, train_sampler, test_sampler, 0, len(train_loader), loss_history, detailed, name_prefix="(0)")

    # This loop repeats for every epoch
    for epoch in range(start_epoch, epochs):

        # This loop repeats for every batch. Tqdm library is used for displaying a progress bar
        for i, multi_batch in enumerate(progressbar := tqdm(train_loader, desc=f'epoch {epoch}', initial = start_iteration, total=len(train_loader) + start_iteration, colour='cyan')):
            
            # Stops training if a given timestamp is reached. Used to stop when the end of a reserved GPU timeslot is reached.
            if isinstance(stop_time, datetime.datetime) and datetime.datetime.now() >= stop_time:
                break

            # calculating the global step so the correct evaluation interval remains over epochs
            step = i + start_iteration + epoch * (len(train_loader)+start_iteration)

            # This loop repeats for every microbatch
            for x, y in zip(multi_batch[0].split(micro_batch_size), multi_batch[1].split(micro_batch_size)):
                x, y = x.to(device), y.to(device) # Move data to the GPU

                # using autocast (Automatic Mixed Precision) for training speedup, if availabe. The backward pass does not have to lie in the context manager
                with autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):  
                    out = model(x) # forward pass through the model
                    B, T, vs = out.shape
                    # Calculating the loss (across all batches) and dividing by the amount of accumulation steps
                    loss = func.cross_entropy(out.view(B * T, vs), y.view(B * T), ignore_index=-1) / accumulation_steps
                scaler.scale(loss).backward() # Backward pass, which calculates gradients for each parameter. Gradients accumulate over multiple backward passes
                batch_loss += loss.detach() # Collecting the losses for display in the progress bar
                del loss, x, y, out

            # At this point a whole batch has accumulated

            # Gradients are clipped, if a value for grad_clip is set
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Optimizer and Gradient scaler perform their steps
            scaler.step(optimizer)
            scaler.update()

            # All accumulated gradients are reset to None, so that a new batch can be trained
            optimizer.zero_grad(set_to_none=True)

            # Since we use a learning rate schedule, the learning rate is updated after the optimizer step. (The learning rate is only used in the optimizer step, hence more frequent updates are not necessary)
            for g in optimizer.param_groups:
                g['lr'] = get_lr(step, epochs*len(train_loader), max_lr, min_lr)

            # Evaluates the current state of the model after a given interval, updates the histories for plotting and saves a checkpoint
            if (step+1) % eval_interval == 0:
                train_loss, test_loss = evaluate(model, device, test_loader, train_set, micro_batch_size, num_workers)
                loss_history['test'].append(test_loss)
                loss_history['train'].append(train_loss)
                progressbar.set_postfix({'running_loss': batch_loss.item(), 'train_loss': train_loss,'test_loss': test_loss})
                save_state(i+1+start_iteration, model, optimizer, scaler, train_sampler, test_sampler, epoch, len(train_loader)+start_iteration, loss_history, detailed)

            # Update the progress bar after a certain interval
            if (step+1) % plot_interval == 0:
                loss_history['running_loss'].append(batch_loss.item())
                progressbar.set_postfix({'running_loss': batch_loss.item(), 'train_loss': train_loss,'test_loss': test_loss})

            # The batch loss is zeroed for the next batch
            batch_loss = torch.tensor(0.0, device=device)

        # This else belongs to a "for" instead of an "if". The else gets executed after the for is finished (as if it was normal code after the loop),
        # but in case of a "break" happening, the code in the else block gets skipped. Therefore we can use this construction to repeats the break in the inner loop again for the outer loop to stop both loops.
        # If no break happens, the else block calls "continue" on the outer loop and therefore skips the second break below.
        else:
            # If no break occurs and the inner loop finishes execution normally, this gets executed at the end of each epoch
            save_state(0, model, optimizer, scaler, train_sampler, test_sampler, epoch, len(train_loader)+start_iteration, loss_history, detailed, name_prefix=f'({epoch+1})')
            start_iteration = 0
            continue # Skips the following line
        break # This is used to propagate a break in the inner loop to the outer loop

# Counts all learnable parameters of a model to give an idea of a model's size
def param_count(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    return params

# This function manages the pretraining process for multiple models by calling other functions in the correct order
def pretrain(models, gpu_num, num_subsets=10, stop_time=None):
    device = torch.device(f'cuda:{gpu_num}') if torch.cuda.is_available() else 'cpu'
    for model in models:
        m, state = Gpt.create_model(model['name'], model['max_epochs'], device, task_name='pretraining', amp_active=torch.cuda.is_available(), kwargs=model['kwargs'])
        print(f"The model has {param_count(m):,} learnable parameters")
        train(m, state, model['max_epochs'], device, num_subsets, stop_time=stop_time, **model['kwargs'])