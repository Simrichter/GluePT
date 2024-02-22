import os
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if not os.path.exists("Plots"):
    os.makedirs("Plots")

file_names = {'cola': 'CoLA', 'stsb': 'STS-B', 'sst2': 'SST-2', 'wnli': 'WNLI', 'rte': 'RTE', 'qnli': 'QNLI',
              'mrpc': 'MRPC', 'qqp': 'QQP', 'mnli': 'MNLI'}
    
def plot_pretraining(models, add_epoch_marks=False, y_min=2.5, y_max=4.5):
    for m in models:
        if "gpt" in m['name']:
            continue
        model_name = m['name']
        epoch = m['max_epochs']
        res=100

        state = torch.load(f'Checkpoints/{model_name}/({epoch}){model_name}.pt', map_location='cpu')
            
        fig, ax1 = plt.subplots(figsize=(8, 5))
            
        loss_history = state['loss_history']
        eval_intervall = loss_history['test_interval']
        y_train = loss_history['train']
        y_test = loss_history['test']
        ax1.plot(range(0, len(y_train)*eval_intervall, eval_intervall), y_train, label='train loss')
        ax1.plot(range(0, len(y_test)*eval_intervall, eval_intervall), y_test, label='validation loss')
        if add_epoch_marks:
            ax1.vlines([i for i in range(len(y_train)*eval_intervall) if i % 34481 == 0], ymin=0.0, ymax=10.0, linestyle=(0, (1, 10)), color="red", linewidth=0.6)
            ax2 = ax1.secondary_xaxis("top", functions = (lambda x: x/34481, lambda x: x*34481))
            ax2.set_xlabel("epochs")
            fig.legend(bbox_to_anchor=(0.965, 0.88), fancybox=True)
        else:
            fig.legend(bbox_to_anchor=(0.98, 0.967), fancybox=True)

        plt.grid(axis='y', color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        fig.tight_layout()
        ax1.set_ylabel('cross-entropy loss')
        ax1.set_xlabel('trained batches')
        

        ax1.set_ylim(y_min, y_max)
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        if not os.path.exists(f"Plots/{model_name}"):
            os.makedirs(f"Plots/{model_name}")
        plt.savefig((f'Plots/{model_name}/pretrain_loss_{model_name}.png'), bbox_inches='tight', dpi=res)
        plt.close()


# This function creates an overview plot over all nine tasks, showing the score development over the pretrain epochs
# add_gpt2 should only be set to true, if the corresponding gpt2 models have been finetuned
def create_overview(models, tasks, add_gpt2=False, detailed=False, select_epochs='all', plot_differences=False):
        
    if detailed:
        create_overview_detailed(models, tasks, add_gpt2, select_epochs)

    freezed = False

    if add_gpt2:
        model_names = [m['name'] for m in models]
        if "small_model" in model_names:
            models.append({"name":"gpt2_small"})
        if "large_model" in model_names:
            models.append({"name":"gpt2_medium"})
    total_data = collect_data(models, tasks, select_epochs, False, freezed)
    plot_overview(models, tasks, total_data, add_gpt2, detailed=False)

    if plot_differences:
        assert len(model_names) == 2, "For plotting differences, exactly two models are required"
        plot_differences(models, tasks, total_data, add_gpt2, detailed = False)

def create_overview_detailed(models, tasks, add_gpt2=False, select_epochs='all'):
        
    freezed = False
            
    if add_gpt2:
        model_names = [m['name'] for m in models]
        if "small_model" in model_names:
            models.append({"name":"gpt2_small"})
        if "large_model" in model_names:
            models.append({"name":"gpt2_medium"})

    total_data = collect_data(models, tasks, select_epochs, True, freezed)
    plot_overview(models, tasks, total_data, add_gpt2, detailed=True)


def collect_data(models, tasks, select_epochs, detailed, freezed):

    print("--------------Collecting data, this may take a while--------------")
    total_data={}
    for m in models:
        model = m['name']
        # creates a data structure holding all necessary values
        total_data[model] = {}
        if "gpt2" in model:
            epochs = [1]
        elif select_epochs=='custom':
            assert 'finetuning_epochs' in m, "!!ERROR!!\nepoch selection strategy 'custom' requires the key 'finetuning_epochs' in the 'models' dictionary"
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
        
        for t in tasks:
            task = t['task_name']
            task_res = {}
            print(f"loading {model}, task: {task}")
            for epoch in epochs:
                if 'gpt2' in model and epoch !=1:
                    continue
                sd = torch.load(f"FinetunedModels/{model}/({epoch}){model}/{'freezed_' if freezed else ''}{task}_({epoch}){model}.pt", map_location='cpu')['score_history']

                keys = sd[0].keys()
                scores = {k: [s[k] for s in sd] for k in keys}
                res = {k: sum(scores[k][-2:])/2 for k in keys}
                task_res[epoch] = res
            total_data[model][task] = task_res
    return total_data
    

def plot_overview(models, tasks, total_data, add_gpt2, detailed):

    resolution = 100
    print("--------------Plotting--------------")
    for m in models:
        model_name = m['name']
        # Do not create an extra plot for gpt2 models
        if "gpt2" in model_name:
            continue
        data = total_data[model_name]

        baselines = {'cola': [0], 'sst2': [50.9], 'stsb': [0, 0], 'mrpc':[68.4, 81.2], 'qqp': [63.2, 53.8], 'mnli': [35.4], 'qnli': [50.5], 'rte': [52.7], 'wnli': [56.3]}

        fig, axs = plt.subplots(3, 3, constrained_layout=True, figsize=(11, 6), dpi=resolution)

        for i, t in enumerate(tasks):
            task = t['task_name']
            axs[i//3, i%3].set_title(file_names[task])
            axs[i//3, i%3].set_ylim(-5, 100)
            axs[i//3, i%3].set_ylabel('validation score', labelpad=-4)
            axs[i//3, i%3].set_xlabel('pretraining epochs', labelpad=-10)
            
            axs[i//3, i%3].yaxis.set_major_locator(ticker.MultipleLocator(20))
            axs[i//3, i%3].yaxis.set_minor_locator(ticker.MultipleLocator(10))
            axs[i//3, i%3].grid(which='minor', axis='y', color='k', linestyle='-', linewidth=0.5, alpha=0.2)
            axs[i//3, i%3].grid(which='major', axis='y', color='k', linestyle='-', linewidth=0.5, alpha=0.5)
            
            epochs = []
            metrics = [*[*data[task].items()][0][1].keys()]
            num_metrics = len([*data[task].items()][0][1].values())
            y = [[] for i in range(num_metrics)]
            for ep, scores in data[task].items():
                epochs.append(ep)
                for j, (metric, res) in enumerate(scores.items()):
                    y[j].append(res*100)
            for k, score in enumerate(y):
                axs[i//3, i%3].plot([e*10 for e in epochs] if detailed else epochs, score, label=metrics[k], color=f'C{k}', marker='x')
                if add_gpt2:
                    axs[i//3, i%3].plot(len(epochs), total_data[f"{'gpt2_small'if model_name=='small_model' else 'gpt2_medium'}"][task][1][metrics[k]]*100, marker='.') #
            for cln, bl in enumerate(baselines[task]):
                axs[i//3, i%3].axhline(y=bl, color=f'C{cln}', linestyle=':')
            if add_gpt2:
                axs[i//3, i%3].set_xticks([*range(len(epochs)+1)], labels=[str(i) for i in epochs]+["GPT2"], rotation='vertical') # range(16)
            else:
                axs[i//3, i%3].set_xticks([*range(len(epochs))], labels=[str(i) for i in epochs], rotation='vertical') # range(16)
            axs[i//3, i%3].legend(loc="lower right")
        
        if not os.path.exists(f"Plots/{model_name}"):
            os.makedirs(f"Plots/{model_name}")
        plt.savefig(f"Plots/{'detailed_' if detailed else ''}{model_name}/overview_{model_name}.png")
        plt.close()

def plot_differences(models, tasks, total_data, add_gpt2, detailed=False):
    fig, axs = plt.subplots(3, 3, constrained_layout=True, figsize=(11, 6), dpi=res)

    for i, t in enumerate(tasks):
        task = t['task_name']
        axs[i//3, i%3].set_title(file_names[task])
        axs[i//3, i%3].set_ylim(-10, 15)
        axs[i//3, i%3].set_ylabel('score difference', labelpad=-4)
        axs[i//3, i%3].set_xlabel('pretraining epochs', labelpad=-10)
        
        axs[i//3, i%3].yaxis.set_major_locator(ticker.MultipleLocator(5))
        axs[i//3, i%3].yaxis.set_minor_locator(ticker.MultipleLocator(2.5))
        axs[i//3, i%3].grid(which='minor', axis='y', color='k', linestyle='-', linewidth=0.5, alpha=0.2)
        axs[i//3, i%3].grid(which='major', axis='y', color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        
        epochs = []
        metrics = [*[*total_data['small_model'][task['task']].items()][0][1].keys()]
        num_metrics = len([*total_data['small_model'][task['task']].items()][0][1].values())
        y = [[] for i in range(num_metrics)]
        for ep, scores in total_data['small_model'][task['task']].items():
            epochs.append(ep)
            for j, (metric, res) in enumerate(scores.items()):
                small=res*100
                large=list(total_data['large_model'][task['task']][ep].values())[j]*100
                y[j].append(large-small)
        for k, score in enumerate(y):
            axs[i//3, i%3].plot(epochs, score, label=metrics[k], color=f'C{k}')
            axs[i//3, i%3].axhline(y=0, color='r', linestyle=':')#, color='r'
            if add_gpt2:
                axs[i//3, i%3].plot(16, total_data['gpt2_medium'][task['task']][1][metrics[k]]*100-total_data['gpt2_small'][task['task']][1][metrics[k]]*100, marker='.') #
        # for cln, bl in enumerate(baselines[task['task']]):
        #     axs[i//3, i%3].axhline(y=bl, color=f'C{cln}', linestyle=':')#, color='r'
        axs[i//3, i%3].set_xticks([*range(17)], labels=[str(i) for i in range(16)]+["GPT2"], rotation='vertical')
        axs[i//3, i%3].legend(loc="lower right")
        if add_gpt2:
            axs[i//3, i%3].set_xticks([*range(17)], labels=[str(i) for i in range(16)]+["GPT2"], rotation='vertical')
        else:
            axs[i//3, i%3].set_xticks([*range(16)], labels=[str(i) for i in range(16)], rotation='vertical')

    plt.savefig((f'Plots/overview_differences.png'))
    plt.close()