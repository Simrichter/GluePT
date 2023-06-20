import torch
import Model.Gpt as gpt
import Model.config as config
import Model.Embeddings as emb

context_dimension = 128#256  # Time
embedding_dimension = 192#384  # feature Channels (Should be multiple of num_heads)
batch_size = 32  # Batch
num_heads = 6  # for multi-head Attention
num_blocks = 6
vocab_size = 50257  # TODO Use 50304 (next multiple of 64) for efficiency ??

use_existing_model = True
training_iterations = 2000
eval_iters = 50
eval_interval = 200
train_test_percentage = 0.9
learning_rate = 5e-4
dropout = 0.2  # TODO implement dropout

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = torch.load('shakespeare_tokenized.pt').to(device)
train_split = data[:int(train_test_percentage * len(data))]
test_split = data[int(train_test_percentage * len(data)):]

parameters = config.params(embedding_dimension, n_heads=num_heads, n_blocks=num_blocks,
                           batchsize=batch_size, context_length=context_dimension, vocab_size=vocab_size,
                           device=device)

model = gpt.GptModel(parameters).to(device)


def get_batch(test=False):
    if test:
        d = test_split
    else:
        d = train_split
    # -2 because len(d) is 1 larger than the last index of d and y needs a shift to the right by 1
    indizes = torch.randint(low=0, high=max(len(d) - context_dimension - 2, 0), size=(batch_size,))
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


def training_loop(iterations):
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate, foreach=False,
                                  fused=True)  # TODO Test if fused is good

    for steps in range(1, iterations+1):
        optimizer.zero_grad(set_to_none=True)
        x, y = get_batch()
        _, loss = model(x, y)
        if steps % eval_interval == 0:
            test_loss = estimate_loss()
            print("Iteration ", steps, " train loss: ", loss.item(), " test loss: ", test_loss.item())
            # print("loss after ", steps, " iterations: ", estimate_loss())
        loss.backward()
        optimizer.step()


def param_count(module=model):
    model_parameters = filter(lambda p: p.requires_grad, module.parameters())
    params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    return params


if use_existing_model:
    model.load_state_dict(torch.load('model_save.pt'))
print("Number of Parameters: ", '{0:,}'.format(param_count()))
#training_loop(training_iterations)
torch.save(model.state_dict(), 'model_save.pt')
s = "First Citizen:\nBefore we proceed any further, hear me speak."
s2 = "First Citizen \nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?"
print(emb.decode(
    model.generate(torch.tensor(emb.encode(s2)).view(1,-1).to(device), 100)[0].tolist()))
