import torch
import Model.Gpt as gpt
import Model.config as config
import Model.Embeddings as emb

model_name = 'reg_shakespeare'

context_dimension = 256#512  # Time
embedding_dimension = 768  # 192#384  # feature Channels (Should be multiple of num_heads)
batch_size = 32  # Batch
num_heads = 12  # for multi-head Attention
num_blocks = 12
vocab_size = 50257
dropout = 0.35 #0.2

train_test_percentage = 0.99

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = torch.load('shakespeare_tokenized.pt', map_location=device)#.to(device)
#data = torch.load('Bookcorpus/bc1.pt')
data = torch.load('shakespeare_tokenized.pt')
#print(data.shape)

train_split = data[:int(train_test_percentage * len(data))]
test_split = data[int(train_test_percentage * len(data)):]

parameters = config.params(embedding_dimension, n_heads=num_heads, n_blocks=num_blocks,
                           batchsize=batch_size, context_length=context_dimension, vocab_size=vocab_size,
                           device=device, dropout=dropout)
if device == 'cuda':
    model = torch.compile(gpt.GptModel(parameters).to(device))
else:
    model = gpt.GptModel(parameters).to(device)

def param_count(module=model):
    model_parameters = filter(lambda p: p.requires_grad, module.parameters())
    params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    return params

#model.load_state_dict(torch.load('model_save.pt'))

state = torch.load('final_{}.pt'.format(model_name))
model.load_state_dict(state['state_dict'])
#optimizer.load_state_dict(state['optimizer'])
#iteration_offset = state['iteration']
#loss_history = state['loss_history']

#model.load_state_dict(torch.load('checkpoint.pt'))
print("Number of Parameters: ", '{0:,}'.format(param_count()))
itrain = torch.randint(low=0, high = len(train_split), size=(4,))
itest = torch.randint(low=0, high = len(test_split), size=(4,))
print(itest[2])
test_strings = [[torch.tensor(emb.encode('\n')),'(newline)'],
                [torch.tensor(emb.encode(' ')), '(whitespace)'],
                #[torch.tensor(emb.encode("usually , he would be tearing around the living room , playing with his toys .")), '(First Sentence in Dataset)'],
                [torch.tensor(emb.encode("First Citizen:\nBefore we proceed any further, hear me speak.")), '(First Sentence in Dataset)'],
                [train_split[itrain[0]:itrain[0]+10],  '(randomly drawn from training set)'] ,
                [train_split[itrain[1]:itrain[1]+10],  '(randomly drawn from training set)'] ,
                [test_split[itest[0]:itest[0]+10], '(randomly drawn from test set)'],
                [test_split[itest[1]:itest[1]+10], '(randomly drawn from test set)'],
                [train_split[itrain[2]:itrain[2]+context_dimension],  '(randomly drawn from training set, full context length)'] ,
                [train_split[itrain[3]:itrain[3]+context_dimension],  '(randomly drawn from training set, full context length)'],
                [test_split[itest[2]:itest[2]+context_dimension], '(randomly drawn from test set, full context length)']
                #[test_split[itest[3]:itest[3]+context_dimension], '(randomly drawn from test set, full context length)']
               ]


model.eval()
for l in test_strings:
    print('Input Sequence\n{}:\n {}\n\n'.format(l[1], emb.decode(l[0].tolist())))
    print('Generated Text:\n')
    print(emb.decode(model.generate(l[0].to(device), 100).tolist()))
    print('\n\n---------------------------------')


#print(emb.decode(model.generate(train_split[start_index:context_dimension-127+start_index], 100).tolist()))
#print(emb.decode(model.generate(train_split[0:1], 100).tolist()))
#print(emb.decode(model.generate(torch.tensor(emb.encode('To be or not to be, that is the '), device=device), 150).tolist()))
#print(emb.decode(model.generate(torch.tensor(emb.encode(' '), device=device), 150).tolist()))
model.train()