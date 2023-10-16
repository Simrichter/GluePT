import math

import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm import tqdm
import Model.config as config
import Model.Gpt as Gpt
import Model.Embeddings as emb
from torch.cuda.amp import GradScaler
from torch import autocast
from torch.nn import functional as func
import Data
import os

context_dimension = 256  # 1024  # Time
embedding_dimension = 768  # 192  # feature Channels (Should be multiple of num_heads)
accumulation_steps = 32  # Used to simulate larger batchsize even though finetuning samples can't be batched
batch_size = 32 // accumulation_steps  # (micro)Batch always results in 1 because finetuning samples have variable length and can't be batched
num_heads = 12  # for multi-head Attention
num_blocks = 12
vocab_size = 50257

model_name = '(14)NEWset_owt_model'  # 'test_model'#'gpt2_owt_model' # '4set_owt_model' #'gpt2'#
use_existing_model = os.path.exists(f'Checkpoints/{model_name}.pt')  # and False#
tasks = ['cola']  ##'stsb'#'sst2''cola''wnli''rte''qnli'
compile_model = False  # compiling does not work well with input of fluctuating length
evaluate_only = False

epochs = 3  # TODO
num_workers = 4
# train_test_percentage = 0.99

eval_interval = 500  # 2000 # 1000 #TODO
plot_interval = 1  # log every step
always_save_checkpoints = True
eval_tolerance = 5e-2

dropout = 0.0  # 1  # 0.35 #0.2
weight_decay = 1e-2
grad_clip = 1.0  # 5.0

learning_rate = 5e-5  # 2.5e-4 #  TODO
min_lr = 0  # 1e-6 #TODO

device = torch.device('cuda:2') if torch.cuda.is_available() else 'cpu'

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
for task in tasks:
    metric = load_metric('glue', task)

    # train_data = load_dataset('glue', task, split='train').map(
    #    lambda x: {'sentence': torch.tensor(emb.encode((x["sentence"])))}, batched=False)  # emb.encode(x["sentence"])
    # test_data = load_dataset('glue', task, split='validation').map(
    #    lambda x: {'sentence': torch.tensor(emb.encode((x["sentence"])))}, batched=False)

    # train_data = Data.ExternalDataset(train_data, x='sentence', y='label')
    # test_data = Data.ExternalDataset(test_data, x='sentence', y='label')
    train_data = Data.FinetuneData(task, 'train')
    test_data = Data.FinetuneData(task, 'validation')

    train_loader = DataLoader(train_data, num_workers=num_workers, batch_size=batch_size, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_data, num_workers=num_workers, batch_size=batch_size, pin_memory=True, shuffle=True)

    parameters = config.params(embedding_dimension, n_heads=num_heads, n_blocks=num_blocks,
                               batchsize=batch_size, context_length=context_dimension, vocab_size=vocab_size,
                               device=device, dropout=dropout, task=task, bias=True)  # TODO bias test

    model = Gpt.GptModel(parameters).to(device)
    if torch.cuda.is_available() and compile_model:
        print('compiling model')
        model = torch.compile(model)

    scaler = GradScaler()

    loss_history = {"train": [], "test": [], "test_interval": eval_interval, "plot_interval": 1}
    score_history = [0]
    if use_existing_model:
        state = torch.load(f'Checkpoints/{model_name}.pt', map_location=device)
        if not (torch.cuda.is_available() and compile_model):
            sd = {k.removeprefix('_orig_mod.'): v for k, v in
                  state["state_dict"].items()}  # remove '_orig_mod.' prefix to allow loading to an uncompiled Model
        else:
            sd = state["state_dict"]
        model.load_state_dict(sd, strict=False)
        print(f"Model successfully loaded\nstarting {task}-finetuning")
    else:
        print(f"No model loaded, untrained model is used")

    # model_params = filter(lambda p: p.requires_grad, model.parameters()) #TODO
    decay_groups = [{'params': [p for p in filter(lambda p: p.requires_grad and p.dim() >= 2, model.parameters())],
                     'weight_decay': weight_decay},
                    {'params': [p for p in filter(lambda p: p.requires_grad and p.dim() < 2, model.parameters())],
                     'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(decay_groups, learning_rate, (0.9, 0.95), weight_decay=weight_decay,
                                  fused=True)  # TODO learning rate mit warmup initialisieren


    def get_lr(it):
        lr_decay_iters = len(train_loader) * epochs
        warmup = lr_decay_iters * 0.002
        if it < warmup:
            return learning_rate * it / warmup
        if it > lr_decay_iters:
            return min_lr
        return (learning_rate - min_lr) * (1 - (it - warmup) / (lr_decay_iters - warmup)) + min_lr


    def evaluate():
        model.eval()
        with torch.no_grad():
            losses = torch.zeros(len(test_loader))
            preds = []
            refs = []
            for k, batch in enumerate(tqdm(test_loader, leave=False)):
                x, y = batch[0], batch[1]
                # print(x)
                if task == "stsb":
                    x2 = torch.cat((x[1], x[0]), dim=1).to(device)  # , torch.tensor([[50256]])
                    x = torch.cat((x[0], x[1]), dim=1).to(device)  # , torch.tensor([[50256]])
                    # print(x)
                    # print(x2)
                elif task == 'wnli' or task == 'rte' or task == 'qnli':
                    x = torch.cat((x[0], torch.tensor([[50256]]), x[1]), dim=1).to(device)
                else:
                    x = x.to(
                        device)  # torch.cat((torch.tensor([[50256]]), x, torch.tensor([[50256]])), dim=1).to(device)
                y = y.to(device)
                if x.shape[1] > 256:  # TODO
                    continue
                out = model(x)
                if task == "stsb":
                    out2 = model(x2)
                    out = (
                                      out + out2) / 2  # Okay since combining results after heads is equivalent to combining before heads
                    del out2, x2  # TODO check if this bricks stsb
                loss = loss_fn(out, y)
                losses[k] = loss.detach()
                del loss, x

                B, T, vs = out.shape
                # print(y)
                if task == "stsb":  # sts-b is a regression task, thus the output is not treated as a probability distribution but as the actual prediction
                    preds.append(out.view(1)),
                else:
                    # preds.append(torch.multinomial(func.softmax(out.view(vs), dim=-1), num_samples=1).item())
                    preds.append(torch.argmax(out).item())
                refs.append(y.item())
        # print(func.softmax(out, dim=-1).item())
        model.train()
        score = metric.compute(predictions=preds, references=refs)
        # print(f"preds:{preds}, refs:{refs}")
        score_history.append(score)
        return losses.mean().detach()


    def save_state(iteration):
        state_to_save = {
            "state_dict": model.state_dict(),
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "loss_history": loss_history,
            "score_history": score_history[1:]
        }
        torch.save(state_to_save, f'FinetunedModels/finetuned_{task}_{model_name}.pt')


    def loss_fn(out, y):
        assert y != -1
        B, T, vs = out.shape
        if task == 'stsb':
            return func.mse_loss(out.view(B * T, vs), y, reduction='mean')
        else:
            return func.cross_entropy(out.view(B * T, vs), y.view(B * T))


    def training_loop():
        batch_loss = 0
        test_loss = torch.tensor(0)

        for epoch in range(epochs):
            for i, batch in enumerate(progressbar := tqdm(train_loader, desc=f'epoch {epoch}')):
                step = i + epoch * len(train_loader)
                # x, y = torch.tensor(batch['sentence'])[None, :].detach().to(device), batch['label'].to(device)
                x, y = batch[0], batch[1]
                if task == "stsb":
                    x2 = torch.cat((x[1], x[0]), dim=1).to(device)  # , torch.tensor([[50256]])
                    x = torch.cat((x[0], x[1]), dim=1).to(device)  # , torch.tensor([[50256]])
                elif task == "wnli" or task == 'rte' or task == 'qnli':
                    x = torch.cat((x[0], torch.tensor([[50256]]), x[1]), dim=1).to(device)  # , torch.tensor([[50256]])
                else:
                    x = x.to(device)
                y = y.to(device)

                # print(f"x:{x}")
                # print(f"y:{y}")
                if x.shape[1] > 256:  # TODO
                    continue
                with autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    out = model(x)
                    if task == "stsb":
                        out2 = model(x2)
                        out = out + out2
                    loss = loss_fn(out, y) / accumulation_steps  # , ignore_index=-1)
                    assert not math.isnan(loss)

                    batch_loss += loss.detach()
                scaler.scale(loss).backward()
                del loss, x, y, out

                if ((step + 1) % accumulation_steps == 0) or (step + 1 == len(train_loader)):
                    if (step) % plot_interval == 0:  # log every step
                        loss_history['train'].append(batch_loss.item())
                        progressbar.set_postfix({'train_loss': batch_loss.item(), 'test_loss': test_loss.item(),
                                                 'score': score_history[-1]})

                    if grad_clip != 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    for g in optimizer.param_groups:
                        g['lr'] = get_lr(step)  # + epoch * len(train_loader)
                    batch_loss = 0

                if (step + 1 - 1) % eval_interval == 0:  # TODO -1?
                    test_loss = evaluate()  # estimate_loss()
                    loss_history['test'].append(test_loss.item())
                    # print("Iteration ", steps, " train loss: ", loss.item(), " test loss: ", test_loss.item())
                    # progressbar.set_postfix({'train_loss': loss.item(), 'test_loss': test_loss.item()})

                    # if test_loss < best_loss + eval_tolerance or always_save_checkpoints:
                    best_loss = test_loss
                    save_state(step + epoch * len(train_loader))
                    # else:
                    # print("test loss got larger, no checkpoint will be saved")
        test_loss = evaluate()
        loss_history['test'].append(test_loss.item())
        if test_loss < best_loss + eval_tolerance or always_save_checkpoints:
            best_loss = test_loss
            save_state(step + epoch * len(train_loader))
        else:
            print("test loss got larger, no checkpoint will be saved")


    if evaluate_only:
        for i in range(5):
            print('starting evaluation')
            test_loss = evaluate()
            print(test_loss)
    else:
        training_loop()
