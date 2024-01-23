import math

import torch
from datasets import load_metric
from torch.utils.data import DataLoader
from tqdm import tqdm
import Model.config as config
import Model.Gpt as Gpt
import Model.Embeddings as emb
from torch.cuda.amp import GradScaler
from torch import autocast
from torch.nn import functional as func
from statistics import mean 
import Data
import os

#Model dimensions (only uncomment when using custom model setups)
#--------------------------------------------------------------------------------------------
#embedding_dimension = 1024#768 # # 192  # feature Channels (Should be multiple of num_heads)
#num_heads = 16#12 #  # for multi-head Attention
#num_blocks = 24#12 #
#vocab_size = 50257
#bias=True
#--------------------------------------------------------------------------------------------

accumulation_steps = 32 # 128  # Used to simulate larger batchsize even though finetuning samples can't be batched #TODO Was 32
batch_size = 32 // accumulation_steps  # 128   # (micro)Batch always results in 1 because finetuning samples have variable length and can't be batched
context_dimension = 256  # 1024  # Time

freeze_model = False

gpu_num=3

compile_model = False  # compiling does not work well with input of fluctuating length
#evaluate_only = False
def get_lr(it, total_iters, max_lr, min_lr):
            # lr_decay_iters = len(train_loader) * epochs
            warmup = total_iters * 0.002
            if it < warmup:
                return max_lr * it / warmup
            if it > total_iters:
                return min_lr
            return (max_lr - min_lr) * (1 - (it - warmup) / (total_iters - warmup)) + min_lr

def prepare_x(x, task_name):
            if task_name == "stsb":
                x2 = torch.cat((x[1], x[0]), dim=1)  # , torch.tensor([[50256]])
                x = torch.cat((x[0], x[1]), dim=1)  # , torch.tensor([[50256]])
                return x, x2
            elif task_name in ['wnli', 'rte', 'qnli', 'mrpc', 'qqp', 'mnli']:
                return torch.cat((x[0], torch.tensor([[50256]]), x[1]), dim=1)
            else:
                return x
            
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
        for k, batch in enumerate(tqdm(test_loader, leave=False)):
            if k >= 5500: # end evaluation on large validation sets early
                break
            x, y = prepare_x(batch[0], task_name).to(device), batch[1]

            y = y.to(device)

            if task_name == "stsb":
                # Inputs longer than the models maximum context length are ignored to prevent a crash
                if x[0].shape[1] > context_dimension or x[1].shape[1] > context_dimension:
                    print("Warning, sample exceeds maximum context length. Skipping")
                    continue
                out = model(x[0])
                out2 = model(x[1])
                # Following the approach in "Improving Language Understanding by Generative Pre-Training", the results are combined to form the output.
                # However they combine the representations before applying the output head.
                # To not alter with the GPT architecture, we combine the final results after the output heads, which is equivalent
                #This is equivalent to the method described in "Improving Language Understanding by Generative Pre-Training"
                out += out2
                del out2
            else:
                # Inputs longer than the models maximum context length are ignored to prevent a crash
                if x.shape[1] > context_dimension:
                    print("Warning, sample exceeds maximum context length. Skipping")
                    continue
                out = model(x)
            loss = loss_fn(out, y, task_name)
            losses.append(loss.item())
            del loss, x

            if task_name == "stsb":  # sts-b is a regression task, thus the output is not treated as a probability distribution but as the actual prediction
                preds.append(out.view(1)),
            else:
                # The conversion of the output probability distribution to a class prediction can be done by using multinomial sampling,
                # but a simple argmax removes randomness and therefore yields better scores

                # preds.append(torch.multinomial(func.softmax(out.view(vs), dim=-1), num_samples=1).item())
                preds.append(torch.argmax(out).item())
            refs.append(y.item())

        score = metric.compute(predictions=preds, references=refs)
        score_history.append(score)
        model.train()
        return torch.tensor(losses).mean().detach()
    
def save_state(iteration, model, model_name, task_name, pretrain_epoch, optimizer, loss_history, score_history):
        state_to_save = {
            "state_dict": model.state_dict(),
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "loss_history": loss_history,
            "score_history": score_history[1:]
        }
        path = os.path.join('FinetunedModels', f'{model_name}')
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f"({pretrain_epoch}){model_name}")
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = f"{path}/TEMP_{'freezed_' if freeze_model else ''}{task_name}_({pretrain_epoch}){model_name}.pt"
        torch.save(state_to_save, file_path)

def finetune_model(model, model_name, device, pretrain_epoch, task_name, epochs=3, eval_interval = 50, weight_decay = 1e-2, grad_clip = 1.0):
    #path = 'Checkpoints/large_model'
    num_workers = 4
    plot_interval = 1  # log every step
    # always_save_checkpoints = True
    # eval_tolerance = 5e-2

    max_lr = 5e-5  #  TODO
    min_lr = 0  # 1e-6 #TODO

    # performance optimizations
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    metric = load_metric('glue', task_name)

    train_data = Data.FinetuneData(task_name, 'train')
    if task_name =='mnli':
        test_data = Data.FinetuneData(task_name, 'validation_matched')#, Data.FinetuneData(task, 'validation_mismatched')]
    else:
        test_data = Data.FinetuneData(task_name, 'validation')

    train_loader = DataLoader(train_data, num_workers=num_workers, batch_size=batch_size, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_data, num_workers=num_workers, batch_size=batch_size, pin_memory=True, shuffle=True)

    scaler = GradScaler()

    loss_history = {"train": [], "test": [], "test_interval": eval_interval, "plot_interval": plot_interval}
    score_history = [0]
    

    decay_groups = [{'params': [p for p in filter(lambda p: p.requires_grad and p.dim() >= 2, model.parameters())],
                        'weight_decay': weight_decay},
                    {'params': [p for p in filter(lambda p: p.requires_grad and p.dim() < 2, model.parameters())],
                        'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(decay_groups, 0, (0.9, 0.95), weight_decay=weight_decay, fused=True)


    batch_loss = 0
    test_loss = 0
    limit = len(train_loader)
    if epochs < 1:
        limit = epochs*len(train_loader)
        epochs = 1
    for epoch in range(epochs):
        for i, batch in enumerate(progressbar := tqdm(train_loader, desc=f"Epoch: {epoch}", position=0, leave=True, dynamic_ncols=True)):
            if i >= limit: # early ending to train for a fraction of an epoch (used with very large datasets)
                break
            step = i + epoch * len(train_loader)
            x, y = prepare_x(batch[0], task_name).to(device), batch[1].to(device)
            # y = y.to(device)

            with autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                if task_name == "stsb":
                    if x[0].shape[1] > context_dimension or x[1].shape[1] > context_dimension:
                        continue
                    out = model(x[0])
                    out2 = model(x[1])
                    out += out2
                    del out2
                else:
                    if x.shape[1] > context_dimension:
                        continue
                    out = model(x)
                loss = loss_fn(out, y) / accumulation_steps
                assert not math.isnan(loss) # stop training if there is something wrong
                batch_loss += loss.detach()

            scaler.scale(loss).backward()
            del loss, x, y, out

            if ((step + 1) % accumulation_steps == 0) or (step + 1 == len(train_loader)):
                if step % plot_interval == 0:  # logs every step when plot_interval is 1
                    loss_history['train'].append(batch_loss.item())
                    progressbar.set_postfix({'train_loss':
                                                batch_loss.item(),
                                                'test_loss': test_loss,
                                                'score': score_history[-1]})

                if ((step+1)//accumulation_steps) % eval_interval == 0:  # TODO -1? 
                    test_loss = evaluate(model, task_name, test_loader, metric, score_history, device)
                    loss_history['test'].append(test_loss.item())

                    progressbar.set_postfix({'train_loss': batch_loss.item(), 'test_loss': test_loss, 'score': score_history[-1]})

                    save_state(step + epoch * len(train_loader), model, model_name, task_name, pretrain_epoch, optimizer, loss_history, score_history)
                
                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                for g in optimizer.param_groups:
                    g['lr'] = get_lr(step, epoch*len(train_loader), max_lr, min_lr)
                batch_loss = 0
   
        test_loss = evaluate(model, task_name, test_loader, metric, score_history, device)
        loss_history['test'].append(test_loss.item())

        save_state(epochs * len(train_loader), model, model_name, task_name, pretrain_epoch, optimizer, loss_history, score_history)
        
        
    
def create_model(model_name, epoch, device, model_type, task, dropout=0.0, compile_model=False,
                 embedding_dimension=384, num_heads=6, num_blocks=6, vocab_size=50257, bias=False):
    if model_type=='small_model':
        parameters = config.small_model(batchsize=batch_size, context_length=context_dimension,
                                            device=device, dropout=dropout, task=task, freeze_model=freeze_model)
    elif model_type=='large_model':
        parameters = config.large_model(batchsize=batch_size, context_length=context_dimension,
                                        device=device, dropout=dropout, task=task, freeze_model=freeze_model)
    elif model_type=='gpt2_small':
        parameters = config.small_model(batchsize=batch_size, context_length=context_dimension,
                                        device=device, dropout=dropout, task=task, freeze_model=freeze_model, use_gpt2=True)
    elif model_type=='gpt2_medium': # GPT2 medium is of equivalent size to our large model. However naming it gpt2_large would be misleading as a large version of GPT2 also exists.
        parameters = config.large_model(batchsize=batch_size, context_length=context_dimension,
                                        device=device, dropout=dropout, task=task, freeze_model=freeze_model, use_gpt2=True)
    else:
        if not all(var in globals() for var in [embedding_dimension, num_heads, num_blocks]):
            print("!!Error\nAttempting to use custom model setup but missing parameters!!")                  
        parameters = config.params(embedding_dimension=embedding_dimension, n_heads=num_heads, n_blocks=num_blocks, batchsize=batch_size, 
                                context_length=context_dimension, vocab_size=vocab_size, device=device,
                                dropout=dropout, task=task, bias=bias, freeze_model=freeze_model, use_gpt2=False)
    
    model = Gpt.GptModel(parameters).to(device)
    if torch.cuda.is_available() and compile_model:
        print('compiling model')
        model = torch.compile(model)
    use_existing_model = os.path.exists(f"Checkpoints/{model_name}/({epoch}){model_name}.pt")
    if use_existing_model:
            state = torch.load(f"Checkpoints/{model_name}/({epoch}){model_name}.pt", map_location=device)

            #This part is used to load a model that has been compiled while pretraining but is now finetuned without compile
            if not (torch.cuda.is_available() and compile_model):
                sd = {k.removeprefix('_orig_mod.'): v for k, v in state["state_dict"].items()}  # remove '_orig_mod.' prefix to allow loading to an uncompiled Model
            else:
                sd = state["state_dict"]
            model.load_state_dict(sd, strict=False)
            print(f"Model ({epoch}){model_name} successfully loaded\nstarting {task}-finetuning")
    elif not parameters.use_gpt2:
        print(f"!!WARNING!!\nNo model loaded\nstarting {task}-finetuning")
    return model

def check_best(model_name, pretrain_epoch, task_name):
     # Checking, if the finetuned model is better or worse than any previous finetuning
    folder = f"FinetunedModels/{model_name}/({pretrain_epoch}){model_name}"
    file_name = f"{'freezed_' if freeze_model else ''}{task_name}_({pretrain_epoch}){model_name}.pt"
    best_path = f"{folder}/{file_name}"
    temp_path= f"{folder}/TEMP_{file_name}"
    rename = True
    if os.path.exists(best_path): # Only if there already exists a finetuned model, a comparison needs to be done
        # Extracting the final validation scores
        best_score = mean([list(dic.values())[-1] for dic in torch.load(best_path, map_location='cpu')["score_history"][1:]])
        temp_score = mean([list(dic.values())[-1] for dic in torch.load(temp_path, map_location='cpu')["score_history"][1:]])
        print(f"best:{best_score}, this:{temp_score}")
        rename = best_score < temp_score
    if rename: # If the "TEMP_" model is better than the existing one, or there is no other model, it gets renamed.
        if os.path.exists(best_path):
            os.remove(best_path)
        os.rename(temp_path, best_path)
        print("New best model found\n")

def finetune(models, tasks, gpu_num):
    device = torch.device(f'cuda:{gpu_num}') if torch.cuda.is_available() else 'cpu'
    for m in models:
        for e in m.get('epochs'):
            for task in tasks:
                for trie in range(task.get('tries', 1)):
                    print(f"try: {trie}")
                    model = create_model(m['name'], e, device, m['type'], **task)
                    finetune_model(model, m['name'], device, e, **task)
                    check_best(m['name'], e, task['task_name'])


models = [
    # {'name':"large_model", 'epochs':[12, 14], 'type':'large_model'},
    # {'name':"small_model", 'epochs': [15], 'type':'small_model'}
    #{'name':"gpt2_small", 'epochs': [1], 'type':'gpt2_small'}
    {'name':"gpt2_medium", 'epochs': [1], 'type':'gpt2_medium'}
]

tasks = [
    {'task_name': 'cola', 'tries':7}
    ,{'task_name': 'stsb', 'tries':5}
    ,{'task_name': 'sst2', 'tries':2} #, 'epochs':3
    ,{'task_name': 'wnli', 'epochs':6, 'dropout':0.2, 'eval_interval':100} #Achtung, war 0.5
    ,{'task_name': 'rte', 'tries':10}
    ,{'task_name': 'qnli', 'epochs':2}
    ,{'task_name': 'mrpc', 'tries':3}
    ,{'task_name': 'qqp', 'epochs':1, 'eval_interval':10000}
    ,{'task_name': 'mnli', 'epochs':1, 'eval_interval':10000}    
        ]