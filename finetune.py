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

context_dimension = 256  # 1024  # Time
embedding_dimension = 768  # 192  # feature Channels (Should be multiple of num_heads)
accumulation_steps = 32  # 2
batch_size = 32 // accumulation_steps  # (micro)Batch
num_heads = 12  # for multi-head Attention
num_blocks = 12
vocab_size = 50257

model_name = '4set_owt_model'
task = 'stsb'  # 'sst2'#cola'#
compile_model = False  # False#
evaluate_only = False

epochs = 3
num_workers = 4
train_test_percentage = 0.99

eval_interval = 500  # 250 # 1000
always_save_checkpoints = True
eval_tolerance = 5e-2

dropout = 0  # 0.35 #0.2
weight_decay = 0
grad_clip = 0.0  # 5.0

learning_rate = 5e-5  # 2.5e-4
min_lr = 1e-6
warmup_iterations = 200

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

metric = load_metric('glue', task)

# train_data = load_dataset('glue', task, split='train').map(
#    lambda x: {'sentence': torch.tensor(emb.encode((x["sentence"])))}, batched=False)  # emb.encode(x["sentence"])
# test_data = load_dataset('glue', task, split='validation').map(
#    lambda x: {'sentence': torch.tensor(emb.encode((x["sentence"])))}, batched=False)

# train_data = Data.ExternalDataset(train_data, x='sentence', y='label')
# test_data = Data.ExternalDataset(test_data, x='sentence', y='label')
train_data = Data.FinetuneData(task, 'train')
test_data = Data.FinetuneData(task, 'validation')

train_loader = DataLoader(train_data, num_workers=num_workers, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, num_workers=num_workers, batch_size=batch_size, shuffle=True)

parameters = config.params(embedding_dimension, n_heads=num_heads, n_blocks=num_blocks,
                           batchsize=batch_size, context_length=context_dimension, vocab_size=vocab_size,
                           device=device, dropout=dropout, task=task)

#######################
# train_set = Data.Dataset(parameters, train=True, train_test_percentage=train_test_percentage)
# test_set = Data.Dataset(parameters, train=False, train_test_percentage=train_test_percentage)
# train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
##########################

model = Gpt.GptModel(parameters).to(device)
if torch.cuda.is_available() and compile_model:
    print('compiling model')
    model = torch.compile(model, dynamic=False)

scaler = GradScaler()

loss_history = {"train": [], "test": [], "test_interval": eval_interval}
score_history = [0]

state = torch.load('final_{}.pt'.format(model_name))
if not (torch.cuda.is_available() and compile_model):
    sd = {k.removeprefix('_orig_mod.'):v for k, v in state["state_dict"].items()}  # remove '_orig_mod.' prefix to allow loading to an uncompiled Model
else:
    sd = state["state_dict"]
model.load_state_dict(sd, strict=False)
print("Model successfully loaded, performing {}-finetuning".format(task))
model_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(model_params, learning_rate, (0.9, 0.95), weight_decay=weight_decay, foreach=True,
                              fused=False)


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
            if task == "stsb":
                x2 = torch.cat((x[1], x[0]), dim=1).to(device)
                x = torch.cat((x[0], x[1]), dim=1).to(device)
                # print(x)
                # print(x2)
            else:
                x = x.to(device)
            y = y.to(device)

            out = model(x)
            if task == "stsb":
                out2 = model(x2)
                out = out + out2  # Okay since combining results after heads is equivalent to combining before heads
                del out2, x2  # TODO check if this bricks stsb
            loss = loss_fn(out, y)
            losses[k] = loss.detach()
            del loss, x

            B, T, vs = out.shape
            # print(y)
            if task == "stsb":
                preds.append(out.view(1)),
            else:
                preds.append(torch.multinomial(func.softmax(out.view(vs), dim=-1), num_samples=1).item())
                # preds.append(torch.argmax(out).item())
            refs.append(y.item())

    model.train()
    if len(set(preds)) <= 1:
        print('preads are constant:', preds)
    if len(set(refs)) <= 1:
        print('refs are constant:', refs)
    score = metric.compute(predictions=preds, references=refs)
    score_history.append(score)
    return losses.mean().detach()


def save_state(iteration, checkpoint=True):
    state_to_save = {
        "state_dict": model.state_dict(),
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        "loss_history": loss_history,
        "score_history": score_history[1:]
    }
    torch.save(state_to_save, 'FinetunedModels/finetuned_{}_{}.pt'.format(task, model_name))


def loss_fn(out, y):
    assert y != -1
    B, T, vs = out.shape
    if task == 'stsb':
        return func.mse_loss(out.view(B * T, vs), y, reduction='sum')
    else:
        return func.cross_entropy(out.view(B * T, vs), y.view(B * T))


def training_loop():
    batch_loss = 0
    test_loss = torch.tensor(0)

    for epoch in range(epochs):
        print('Epoch: ', epoch)
        for i, batch in enumerate(progressbar := tqdm(train_loader)):
            step = i + epoch * len(train_loader)
            # x, y = torch.tensor(batch['sentence'])[None, :].detach().to(device), batch['label'].to(device)
            x, y = batch[0], batch[1]
            if task == "stsb":
                x2 = torch.cat((x[1], x[0]), dim=1).to(device)
                x = torch.cat((x[0], x[1]), dim=1).to(device)
            else:
                x = x.to(device)
            y = y.to(device)

            with autocast(device_type='cuda', dtype=torch.float16, enabled=False):

                out = model(x)
                if task == "stsb":
                    out2 = model(x2)
                    out = out + out2
                loss = loss_fn(out, y) / accumulation_steps  # , ignore_index=-1)
                assert not math.isnan(loss)
                del x, y, out

                batch_loss += loss.detach()
            scaler.scale(loss).backward()
            del loss

            if ((step + 1) % accumulation_steps == 0) or (step + 1 == len(train_loader)):
                if (step) % 1 == 0:  # not 10 to log every step
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
                    g['lr'] = get_lr(step + epoch * len(train_loader))
                batch_loss = 0

            if (step + 1) % eval_interval == 0:
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
