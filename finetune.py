import math
import warnings

import torch
from datasets import load_metric
from torch.utils.data import DataLoader
from tqdm import tqdm
import Model.Gpt as Gpt
from torch.cuda.amp import GradScaler
from torch import autocast
from torch.nn import functional as func
from statistics import mean 
import Data
import os

def get_lr(it, total_iters, max_lr, min_lr):
            warmup = total_iters * 0.002
            if it < warmup:
                return max_lr * it / warmup
            if it > total_iters:
                return min_lr
            return (max_lr - min_lr) * (1 - (it - warmup) / (total_iters - warmup)) + min_lr

def prepare_x(x, task_name):
            if task_name == "stsb":
                x2 = torch.cat((x[1], x[0]), dim=1)
                x = torch.cat((x[0], x[1]), dim=1)
                return x, x2
            elif task_name in ['wnli', 'rte', 'qnli', 'mrpc', 'qqp', 'mnli']:
                return torch.cat((x[0], x[1]), dim=1)
            else:
                return x
            
# This function calculates the loss by applying the correct loss function to the given output "out" and the correct labels "y"
def loss_fn(out, y, task_name):
            assert y != -1 # Some GLUE tasks have test labels held secret. Finetuning on secret (-1) labels is prevented
            B, T, vs = out.shape
            if task_name == 'stsb':
                return func.mse_loss(out.view(B * T, vs), y, reduction='mean')
            else:
                return func.cross_entropy(out.view(B * T, vs), y.view(B * T))
            
def evaluate(model, task_name, test_loader, metric, score_history, device):
    model.eval()
    with torch.no_grad():
        losses = []
        # preds and refs are used to collect all predictions with their correct references to calculate the metric after the evaluation run
        preds = []
        refs = []
        for k, batch in enumerate(tqdm(test_loader, leave=False, total=min(len(test_loader), 5500))):
            if k >= 5500: # end evaluation on large validation sets early (On GLUE this affects only QQP and MNLI)
                break
            x, y = prepare_x(batch[0], task_name), batch[1].to(device)

            # Special treatment because in stsb the samples are passed through the model in both possible orders
            if task_name == "stsb":
                out = model(x[0].to(device))
                out2 = model(x[1].to(device))
                # Following the approach in "Improving Language Understanding by Generative Pre-Training", the results are combined to form the output.
                # However they combine the representations before applying the output head.
                # To not alter with the GPT architecture, we combine the final results after the output heads, which is equivalent.
                out += out2
                del out2
            else:
                out = model(x.to(device))
            loss = loss_fn(out, y, task_name)
            losses.append(loss.item())
            del loss, x

            if task_name == "stsb":  # sts-b is a regression task, thus the output is not treated as a probability distribution but as the actual prediction
                preds.append(out.view(1)),
            else:
                # The conversion of the output probability distribution to a class prediction can be done by using multinomial sampling,
                # but a simple argmax removes randomness and therefore generally yields better scores
                preds.append(torch.argmax(out).item())
            refs.append(y.item())

        score = metric.compute(predictions=preds, references=refs)
        score_history.append(score)
        model.train()
        return torch.tensor(losses).mean().detach()
    
def save_state(iteration, model, task_name, pretrain_epoch, optimizer, loss_history, score_history):
        state_to_save = {
            "state_dict": model.state_dict(),
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "loss_history": loss_history,
            "score_history": score_history[1:] # exclude the first entry, since it is the default value "0" for displaying
        }
        path = os.path.join('FinetunedModels', f'{model.name}')
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f"({pretrain_epoch}){model.name}")
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = f"{path}/TEMP_{'freezed_' if model.freeze else ''}{task_name}_({pretrain_epoch}){model.name}.pt"
        torch.save(state_to_save, file_path)

def finetune_model(model, device, pretrain_epoch, task_name, batch_size=32, epochs=3, eval_interval = 50, plot_interval = 1, weight_decay = 1e-2, grad_clip = 1.0, max_lr = 5e-5, min_lr = 0, num_workers = 3, **kwargs):

    # Because the samples differ in length, we do not batch multiple samples together, but perform an individual forward pass for each sample. The use of gradient accumulation results in a training behaviour as if training happened on actual batches
    micro_batch_size = 1
    accumulation_steps = batch_size 

    # performance optimizations
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Load_metric gives a warning, since it's functionality will be moved to a new library "evaluate".
    # However we still want to use load_metric. Therefore the warning is supressed.
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        # Loads the correct metrics for the specific task
        metric = load_metric('glue', task_name, trust_remote_code=True)

    # Loads the dataset for training
    train_data = Data.FinetuneData(task_name, 'train')
    
    # Loads the dataset for validation
    # For mnli we use only the matched validation set for evaluation, as our results show that the scores on matched and mismatched sets are very similar.
    # Therefore we consider the mismatched dataset only for the actual evaluation on the test datasets
    if task_name =='mnli':
        test_data = Data.FinetuneData(task_name, 'validation_matched')
    else:
        test_data = Data.FinetuneData(task_name, 'validation')

    # Creating the data loaders
    train_loader = DataLoader(train_data, num_workers=num_workers, batch_size=micro_batch_size, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_data, num_workers=num_workers, batch_size=micro_batch_size, pin_memory=True, shuffle=True)

    # Parameters are placed in two groups, because weight decay is not applied to bias or gain weights
    decay_groups = [{'params': [p for p in filter(lambda p: p.requires_grad and p.dim() >= 2, model.parameters())], 'weight_decay': weight_decay},
                    {'params': [p for p in filter(lambda p: p.requires_grad and p.dim() < 2, model.parameters())], 'weight_decay': 0.0}]
    
    # Instantiates the optimizer and uses the fused implementation when on a GPU since it is faster
    optimizer = torch.optim.AdamW(decay_groups, 0, (0.9, 0.95), weight_decay=weight_decay, fused=True)

    # Instantiates the gradient scaler (but it is only active when cuda is available)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # A dictionary storing all values throughout training. Used for plotting
    loss_history = {"train": [], "test": [], "test_interval": eval_interval, "plot_interval": plot_interval}

    # A list storing the scores. Used for plotting. The initial value "0" is only used for display in the progress bar
    score_history = [0]

    # Variables used for displaying in the progress bar
    batch_loss = 0
    test_loss = torch.tensor(0)

    # If a float between 0 and 1 is given as the number of epochs, this code calculates the iteration, at which the finetuning should be terminated
    # This allows to train for a fraction of an epoch
    limit = len(train_loader)
    if epochs < 1 and epochs >=0:
        limit = epochs*len(train_loader)
        epochs = 1 # Set epochs to the next larger integer value to start training
    elif not isinstance(epochs, int):
        print("Error, epoch must be either a float in the intervall [0, 1), or an integer")
    
    # This loop repeats for every epoch
    for epoch in range(epochs):
        
        # This loop repeats for every micro_batch. Tqdm library is used for displaying a progress bar
        for i, batch in enumerate(progressbar := tqdm(train_loader, desc=f"Epoch: {epoch}", position=0, leave=True, dynamic_ncols=True, colour='cyan')):
            if i >= limit: # early ending to train for a fraction of an epoch (used with very large datasets)
                break

            # calculating the global step so the correct evaluation interval remains over epochs
            step = i + epoch * len(train_loader)

            # The input is prepared for the forward pass
            x, y = prepare_x(batch[0], task_name), batch[1].to(device)

            # using autocast (Automatic Mixed Precision) for training speedup, if availabe. The backward pass does not have to lie in the context manager
            with autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                if task_name == "stsb":
                    out = model(x[0].to(device))
                    out2 = model(x[1].to(device))
                    out += out2
                    del out2
                else:
                    out = model(x.to(device))
                loss = loss_fn(out, y, task_name) / accumulation_steps
                # stop training if there is something wrong
                assert not math.isnan(loss), f"Encountered a NaN loss, aborting the {task_name}-finetuning"

                batch_loss += loss.item() # Collecting the losses for display in the progress bar

            # Backward pass, which calculates gradients for each parameter. Gradients accumulate over multiple backward passes
            scaler.scale(loss).backward()
            del loss, x, y, out

            # If enough micro_batches have accumulated, or the training is at the last epoch, a batch has accumulated
            if ((step + 1) % accumulation_steps == 0) or (step + 1 == len(train_loader)):
                
                # logs the total loss of the accumulated batch and updates the progressbar
                if step % plot_interval == 0:  # logs every step when plot_interval is 1
                    loss_history['train'].append(batch_loss)
                    progressbar.set_postfix({'train_loss': batch_loss, 'test_loss': test_loss.item(), 'score': score_history[-1]})

                # Evaluates the current state of the model after a given interval, updates the histories for plotting and saves a checkpoint
                if ((step+1)//accumulation_steps) % eval_interval == 0: 
                    test_loss = evaluate(model, task_name, test_loader, metric, score_history, device)
                    loss_history['test'].append(test_loss.item())
                    progressbar.set_postfix({'train_loss': batch_loss, 'test_loss': test_loss.item(), 'score': score_history[-1]})
                    save_state(step + epoch * len(train_loader), model, task_name, pretrain_epoch, optimizer, loss_history, score_history)
                
                # Gradients are clipped, if a value for grad_clip is set
                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                # optimizer and gradient scaler perform a step
                scaler.step(optimizer)
                scaler.update()

                # All accumulated gradients are reset to None, so that a new batch can be trained
                optimizer.zero_grad(set_to_none=True)

                # Since we use a learning rate schedule, the learning rate is updated after the optimizer step. (The learning rate is only used in the optimizer step, hence more frequent updates are not necessary)
                for g in optimizer.param_groups:
                    g['lr'] = get_lr(step, epochs*len(train_loader), max_lr, min_lr)
                batch_loss = 0

    # After finetuning is finished, a final evaluation is performed
    test_loss = evaluate(model, task_name, test_loader, metric, score_history, device)
    loss_history['test'].append(test_loss.item())

    save_state(epochs * len(train_loader), model, task_name, pretrain_epoch, optimizer, loss_history, score_history)
        
# This function checks if the currently trained checkpoint (temp) is better than the previous best (best).
# If that is the case or no previous checkpoint exists, the current checkpoint is renamed
def check_best(model_name, pretrain_epoch, task_name, freeze_model=False):
     # Checking, if the finetuned model is better or worse than any previous finetuning
    folder = f"FinetunedModels/{model_name}/({pretrain_epoch}){model_name}"
    file_name = f"{'freezed_' if freeze_model else ''}{task_name}_({pretrain_epoch}){model_name}.pt"
    best_path = f"{folder}/{file_name}"
    temp_path= f"{folder}/TEMP_{file_name}"
    rename = True
    if os.path.exists(best_path): # Only if there already exists a finetuned model, a comparison needs to be done
        # Extracting the final validation scores and comparing the mean over the last 3 validation scores
        best_score = mean([list(dic.values())[-1] for dic in torch.load(best_path, map_location='cpu')["score_history"][-3:]])
        temp_score = mean([list(dic.values())[-1] for dic in torch.load(temp_path, map_location='cpu')["score_history"][-3:]])
        print(f"best:{best_score}, this:{temp_score}")
        rename = best_score < temp_score
    if rename: # If the "TEMP_" model is better than the existing one, or there is no other model, it gets renamed.
        if os.path.exists(best_path):
            os.remove(best_path)
        os.rename(temp_path, best_path)
        print("New best model found\n")

def finetune(models, tasks, gpu_num, select_epochs='last'):
    device = torch.device(f'cuda:{gpu_num}') if torch.cuda.is_available() else 'cpu'
    for m in models:
        finetune_detailed = m.get("detailed", False)
        if select_epochs=='custom':
            if not 'finetuning_epochs' in m:
                print("!!ERROR!!\nepoch selection strategy 'custom' requires the key 'finetuning_epochs' in the 'models' dictionary")
                return
            epochs=m['finetuning_epochs']
        elif select_epochs=='all':
            epochs = [*range(m['max_epochs']+1)]
            if finetune_detailed: # If detailed mode is active, add sub-epochs
                sub_epochs = [(int(i/10) if isinstance(i/10, float) and (i/10).is_integer() else i/10) for i in range(min(2, m['max_epochs'])*10+1) if i not in [0, 10, 20]]
                epochs = sorted(epochs+sub_epochs)
        elif select_epochs=='last':
            epochs = [m['max_epochs']]
        else:
            print(f"!!ERROR!!\nepoch selection strategy '{select_epochs}' is unknown.\n Choose one of 'custom', 'all', 'last'")
            return
        
        for e in epochs:
            for task in tasks:
                for trie in range(task.get('tries', 1)):
                    print(f"{task['task_name']} try: {trie}")
                    model, _ = Gpt.create_model(m['name'], e, device, compile_model=False, **task)
                    finetune_model(model, device, e, **task)
                    check_best(m['name'], e, task['task_name'], model.freeze)