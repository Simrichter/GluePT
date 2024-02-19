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
import shutil

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


models = [
    # {'name':"large_model", 'epochs':[12, 14], 'type':'large_model'},
    # {'name':"small_model", 'epochs': [15], 'type':'small_model'}
    #{'name':"gpt2_small", 'epochs': [1], 'type':'gpt2_small'}
    {'name':"gpt2_medium", 'epochs': [1], 'type':'gpt2_medium'}
    
]

tasks = [
    {'task': 'cola', 'tries':7}
    ,{'task': 'stsb', 'tries':5}
    ,{'task': 'sst2', 'epochs':3, 'tries':2}
    ,{'task': 'wnli', 'epochs':6, 'dropout':0.2, 'eval_interval':100} #Achtung, war 0.5
    ,{'task': 'rte'}
    ,{'task': 'qnli', 'epochs':2, 'tries':1}
    ,{'task': 'mrpc', 'tries':3}
    ,{'task': 'qqp', 'epochs':1, 'tries':1, 'eval_interval':10000}
    ,{'task': 'mnli', 'epochs':1, 'tries':1, 'eval_interval':10000}    
        ]
compile_model = False  # compiling does not work well with input of fluctuating length
#evaluate_only = False


def finetune(model_name, pretrain_epoch, model_type, task, epochs=3, tries=10, dropout=0.0, eval_interval = 50, weight_decay = 1e-2):
    #path = 'Checkpoints/large_model'
    num_workers = 4
    plot_interval = 1  # log every step
    always_save_checkpoints = True
    eval_tolerance = 5e-2

    grad_clip = 1.0

    learning_rate = 5e-5  #  TODO
    min_lr = 0  # 1e-6 #TODO

    use_existing_model = os.path.exists(f"Checkpoints/{model_name}/({pretrain_epoch}){model_name}.pt")  # and False#
    device = torch.device(f'cuda:{gpu_num}') if torch.cuda.is_available() else 'cpu'

    # performance optimizations
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    metric = load_metric('glue', task)

    train_data = Data.FinetuneData(task, 'train')
    if task =='mnli':
        test_data = Data.FinetuneData(task, 'validation_matched')#, Data.FinetuneData(task, 'validation_mismatched')]
    else:
        test_data = Data.FinetuneData(task, 'validation')

    train_loader = DataLoader(train_data, num_workers=num_workers, batch_size=batch_size, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_data, num_workers=num_workers, batch_size=batch_size, pin_memory=True, shuffle=True)
    
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
        if not all(var in globals() for var in [embedding_dimension, num_heads, num_blocks, vocab_size]):
            print("!!Error\nAttempting to use custom model setup but missing parameters!!")                  
        parameters = config.params(embedding_dimension, n_heads=num_heads, n_blocks=num_blocks, batchsize=batch_size, 
                                   context_length=context_dimension, vocab_size=vocab_size, device=device,
                                   dropout=dropout, task=task, bias=bias, freeze_model=freeze_model, use_gpt2='gpt2')
    
    for trie in range(tries):
        model = Gpt.GptModel(parameters).to(device)
        if torch.cuda.is_available() and compile_model:
            print('compiling model')
            model = torch.compile(model)

        scaler = GradScaler()

        loss_history = {"train": [], "test": [], "test_interval": eval_interval, "plot_interval": plot_interval}
        score_history = [0]
        if use_existing_model:
            state = torch.load(f"Checkpoints/{model_name}/({pretrain_epoch}){model_name}.pt", map_location=device)

            #This part is used to load a model that has been compiled while pretraining but is now finetuned without compile
            if not (torch.cuda.is_available() and compile_model):
                sd = {k.removeprefix('_orig_mod.'): v for k, v in state["state_dict"].items()}  # remove '_orig_mod.' prefix to allow loading to an uncompiled Model
            else:
                sd = state["state_dict"]
            model.load_state_dict(sd, strict=False)
            print(f"Model ({pretrain_epoch}){model_name} successfully loaded\nstarting {task}-finetuning")
        elif not parameters.use_gpt2:
            print(f"!!WARNING!!\nNo model loaded\nstarting {task}-finetuning")
            # print(f"Checkpoints/{model_name}/({pretrain_epoch}){model_name}.pt")

        decay_groups = [{'params': [p for p in filter(lambda p: p.requires_grad and p.dim() >= 2, model.parameters())],
                         'weight_decay': weight_decay},
                        {'params': [p for p in filter(lambda p: p.requires_grad and p.dim() < 2, model.parameters())],
                         'weight_decay': 0.0}]
        optimizer = torch.optim.AdamW(decay_groups, 0, (0.9, 0.95), weight_decay=weight_decay, fused=True)

        def get_lr(it):
            lr_decay_iters = len(train_loader) * epochs
            warmup = lr_decay_iters * 0.002
            if it < warmup:
                return learning_rate * it / warmup
            if it > lr_decay_iters:
                return min_lr
            return (learning_rate - min_lr) * (1 - (it - warmup) / (lr_decay_iters - warmup)) + min_lr

        def prepare_x(x):
            if task == "stsb":
                x2 = torch.cat((x[1], x[0]), dim=1).to(device)  # , torch.tensor([[50256]])
                x = torch.cat((x[0], x[1]), dim=1).to(device)  # , torch.tensor([[50256]])
                return x, x2
            elif task in ['wnli', 'rte', 'qnli', 'mrpc', 'qqp', 'mnli']:
                return torch.cat((x[0], torch.tensor([[50256]]), x[1]), dim=1).to(device)
            else:
                return x.to(device)

        def evaluate():
            model.eval()
            with torch.no_grad():
                losses = torch.zeros(len(test_loader))
                # preds and refs are used to collect all predictions with their correct references to calculate the metric after the evaluation run
                preds = []
                refs = []
                for k, batch in enumerate(tqdm(test_loader, leave=False)):
                    if k >= 5500: # end evaluation on large validation sets early
                        break
                    x, y = prepare_x(batch[0]), batch[1]

                    y = y.to(device)

                    if task == "stsb":
                        if x[0].shape[1] > context_dimension or x[1].shape[1] > context_dimension:
                            continue # Inputs longer than the models maximum context length are ignored to prevent a crash
                        out = model(x[0])
                        out2 = model(x[1])

                        # Combining results after the heads have been applied is equivalent to combining them
                        # before applying a single head. Thus, this is equivalent to the method described in "Improving Language Understanding by Generative Pre-Training"
                        out += out2
                        del out2
                    else:
                        if x.shape[1] > context_dimension:
                            continue
                        out = model(x)
                    loss = loss_fn(out, y)
                    losses[k] = loss.detach()
                    del loss, x

                    B, T, vs = out.shape
                    if task == "stsb":  # sts-b is a regression task, thus the output is not treated as a probability distribution but as the actual prediction
                        preds.append(out.view(1)),
                    else:
                        # The conversion of the output probability distribution to a class prediction can be done by using multinomial sampling,
                        # but a simple argmax removes randomness, doesn't need softmax and yields better scores

                        # preds.append(torch.multinomial(func.softmax(out.view(vs), dim=-1), num_samples=1).item())
                        preds.append(torch.argmax(out).item())

                    refs.append(y.item())
            score = metric.compute(predictions=preds, references=refs)
            score_history.append(score)
            model.train()
            return losses.mean().detach()

        def save_state(iteration):
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
            file_path = f"{path}/{'TEMP_' if tries>1 else ''}{'freezed_' if freeze_model else ''}{task}_({pretrain_epoch}){model_name}.pt"
            torch.save(state_to_save, file_path)

        def loss_fn(out, y):
            assert y != -1 # Some GLUE tasks have test labels held secret. Finetuning on secret (-1) labels is prevented
            B, T, vs = out.shape
            if task == 'stsb':
                return func.mse_loss(out.view(B * T, vs), y, reduction='mean')
            else:
                return func.cross_entropy(out.view(B * T, vs), y.view(B * T))

        #def training_loop():
        batch_loss = 0
        test_loss = 0
        limit = len(train_loader)
        if epochs < 1:
            limit = epochs*len(train_loader)
            epochs = 1
        for epoch in range(epochs):
            for i, batch in enumerate(progressbar := tqdm(train_loader, desc=f"Try: {trie}, epoch: {epoch}", position=0, leave=True, dynamic_ncols=True)):
                if i >= limit: # early ending to train for a fraction of an epoch (used with very large datasets)
                    break
                step = i + epoch * len(train_loader)
                x, y = prepare_x(batch[0]), batch[1]
                y = y.to(device)



                with autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    if task == "stsb":
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
                        test_loss = evaluate()
                        loss_history['test'].append(test_loss.item())

                        progressbar.set_postfix({'train_loss': batch_loss.item(), 'test_loss': test_loss, 'score': score_history[-1]})

                        save_state(step + epoch * len(train_loader))
                    
                    if grad_clip != 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    for g in optimizer.param_groups:
                        g['lr'] = get_lr(step)
                    batch_loss = 0

                
        test_loss = evaluate()
        loss_history['test'].append(test_loss.item())

        save_state(epochs * len(train_loader))
        
        # Checking, if the finetuned model is better or worse than any previous finetuning
        folder = f"FinetunedModels/{model_name}/({pretrain_epoch}){model_name}"
        file_name = f"{'freezed_' if freeze_model else ''}{task}_({pretrain_epoch}){model_name}.pt"
        best_path = f"{folder}/{file_name}"
        temp= f"{folder}/TEMP_{file_name}"
        rename = True
        if os.path.exists(best_path):
            k=2 # Averaging the top k scores to be more robust against outliers
            best_score = mean([list(dic.values())[-1] for dic in torch.load(best_path, map_location='cpu')["score_history"][1:]])
            #sum([sum(dic.values())/len(dic.values()) for dic in torch.load(best_path, map_location='cpu')["score_history"][1:]][-k:])/k
            
            this_score = mean([list(dic.values())[-1] for dic in score_history[1:]])
            #sum([sum(dic.values())/len(dic.values()) for dic in score_history[1:]][-k:])/k
            print(f"best:{best_score}, this:{this_score}")
            rename = tries > 1 and best_score < this_score
        if rename:
            #this_path = f"{folder}/{temp}"
            if os.path.exists(best_path):
                os.remove(best_path)
            os.rename(temp, best_path)
            print("New best model found\n")
        
    if tries>1:
        file_path = f"FinetunedModels/{model_name}/({pretrain_epoch}){model_name}/TEMP_{'freezed_' if freeze_model else ''}{task}_({pretrain_epoch}){model_name}.pt"
        if os.path.exists(file_path):
            os.remove(file_path)
    

for m in models:
    for e in m.get('epochs'):
        for benchmark in tasks:
            t = benchmark['task']
            finetune(m['name'], e, m['type'], **benchmark)
        
        #if 'epochs' in benchmark.keys():
        #    finetune(name, t, benchmark['epochs'])
        #else:
        #    finetune(name, t)
