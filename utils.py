import math
import matplotlib.pyplot as plt

import torch


def estimate_size():
    context_dimension = 256  # Time
    embedding_dimension = 768  # 192#384  # feature Channels (Should be multiple of num_heads)
    # batch_size = 64  # Batch
    # num_heads = 6  # for multi-head Attention
    num_blocks = 12
    vocab_size = 50257

    fixed_positional_encoding = True
    loaded_embeddings = True
    weight_tying = True

    params = 0

    if not loaded_embeddings:
        params += vocab_size * embedding_dimension
    if not fixed_positional_encoding:
        params += context_dimension * embedding_dimension
    if not weight_tying:
        params += embedding_dimension * vocab_size

    params_block = 0
    params_block += 4 * (embedding_dimension ** 2) + 4 * embedding_dimension
    params_block += 8 * (embedding_dimension ** 2) + 5 * embedding_dimension  # param_count of feed-forward net

    params += num_blocks * params_block

    print("Number of Parameters: ", '{0:,}'.format(params))


def plot_loss():
    model_name='shakespeare'
    state = torch.load('final_{}.pt'.format(model_name))
    #state = torch.load('shakespeare.pt')
    loss_history = state['loss_history']
    y_train = [y.cpu().detach().numpy() for y in loss_history['train']]
    y_test = [y.cpu().detach().numpy() for y in loss_history['test']]
    eval_intervall = loss_history['test_interval']
    plt.yscale("log")
    plt.plot(range(len(y_train)), y_train, label='train')
    plt.plot(range(250, len(y_test)*eval_intervall+250, eval_intervall), y_test, label='test')
    plt.legend()
    plt.savefig('final_{}.png'.format(model_name), bbox_inches='tight')
    plt.show()

# plot_loss()

estimate_size()