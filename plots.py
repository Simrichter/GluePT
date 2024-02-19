import os
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if not os.path.exists("Plots"):
    os.makedirs("Plots")

def plot_pretraining(models, add_epoch_marks=False):
    for m in models:
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
        

        ax1.set_ylim(2.5, 4.5)
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        if not os.path.exists(f"Plots/{model_name}"):
            os.makedirs(f"Plots/{model_name}")
        plt.savefig((f'Plots/{model_name}/loss_{model_name}.png'), bbox_inches='tight', dpi=res)
        plt.close()


def plot_finetuning(models, tasks, select_epochs='last'):
    freezed = False
    res=100

    for m in models:
        model_name=m['name']
        if select_epochs=='custom':
            if not 'finetuning_epochs' in m:
                print("!!ERROR!!\nepoch selection strategy 'custom' requires the key 'finetuning_epochs' in the 'models' dictionary")
                return
            epochs=m['finetuning_epochs']
        elif select_epochs=='all':
            epochs = [*range(m['max_epochs'])]
        elif select_epochs=='last':
            epochs = [m['max_epochs']]
        else:
            print(f"!!ERROR!!\nepoch selection strategy '{select_epochs}' is unknown.\n Choose one of 'custom', 'all', 'last'")
            return
        for epoch in epochs:
            for t in tasks:
                task = t['task_name']
                state = torch.load(f"FinetunedModels/{model_name}/({epoch}){model_name}/{'freezed_' if freezed else''}{task}_({epoch}){model_name}.pt", map_location='cpu')
                    
                fig, ax1 = plt.subplots()
                    
                loss_history = state['loss_history']
                eval_intervall = loss_history['test_interval']
                plot_intervall = loss_history["plot_interval"]*32
                y_train = loss_history['train']
                y_test = loss_history['test']

                ax1.set_ylabel("cross-entropy loss")
                ax1.set_xlabel('trained batches')

                plt.grid(axis='y', color='k', linestyle='-', linewidth=0.5, alpha=0.5)
                fig.tight_layout()
                
                ax1.plot(range(0, len(y_train)*plot_intervall, plot_intervall), y_train, label='train loss')
                ax1.plot(range(0, len(y_test)*eval_intervall, eval_intervall), y_test, label='validation loss')
                ax2 = ax1.twinx()
                ax2.set_ylabel('validation score')
                score_history = state['score_history']
                #print(len(score_history))
                metrics = [*score_history[0].keys()]
                y_scores = {m: [ e[m]*100 for e in score_history] for m in metrics}
                #print(y_scores)
                for m in metrics:
                    ax2.plot(range(0, len(y_scores[m])*eval_intervall, eval_intervall), y_scores[m], label=m)
                fig.legend(bbox_to_anchor=(0.98, 0.967), fancybox=True)

                if not os.path.exists(f"Plots/{model_name}"):
                    os.makedirs(f"Plots/{model_name}")
                plt.savefig((f"Plots/{model_name}/{'freezed_' if freezed else''}{task}_{model_name}.png"), bbox_inches='tight', dpi=res)
                plt.close()

# This function creates an overview plot over all nine tasks, showing the score development over the pretrain epochs
# add_gpt2 should only be set to true, if the corresponding gpt2 models have been finetuned
def create_overview(models, tasks, add_gpt2=False, detailed=False, select_epochs='all'):

    if detailed:
        create_overview_detailed(models, tasks, add_gpt2, select_epochs)

    freezed = False
    
    print("--------------Collecting data, this may take a while--------------")
    total_data={}

    if add_gpt2:
        model_names = [m['name'] for m in models]
        if "small_model" in model_names:
            models.append({"name":"gpt2_small"})
        if "large_model" in model_names:
            models.append({"name":"gpt2_medium"})

    
    for m in models:
        model = m['name']
        # creates a data structure holding all necessary values
        total_data[model] = {}
        if "gpt2" in model:
            epochs = [1]
        elif select_epochs=='custom':
            if not 'finetuning_epochs' in m:
                print("!!ERROR!!\nepoch selection strategy 'custom' requires the key 'finetuning_epochs' in the 'models' dictionary")
                return
            epochs=m['finetuning_epochs']
        elif select_epochs=='all':
            epochs = [*range(m['max_epochs']+1)]
        elif select_epochs=='last':
            epochs = [m['max_epochs']]
        else:
            print(f"!!ERROR!!\nepoch selection strategy '{select_epochs}' is unknown.\n Choose one of 'custom', 'all', 'last'")
            return
        
        for t in tasks:
            task = t['task_name']
            task_res = {}
            for epoch in epochs:
                print(f"loading {model}, task: {task}, epoch: {epoch}")
                if 'gpt2' in model and epoch !=1:
                    continue
                sd = torch.load(f"FinetunedModels/{model}/({epoch}){model}/{'freezed_' if freezed else ''}{task}_({epoch}){model}.pt", map_location='cpu')['score_history']

                keys = sd[0].keys()
                scores = {k: [s[k] for s in sd] for k in keys}
                res = {k: sum(scores[k][-2:])/2 for k in keys}
                task_res[epoch] = res
            total_data[model][task] = task_res
    # torch.save(total_data, "Plots/total_data.pt")

    print("--------------Plotting--------------")
    for m in models:
        model_name = m['name']
        # Do not create an extra plot for gpt2 models
        if "gpt2" in model_name:
            continue
        data = total_data[model_name]

        baselines = {
            'cola': [0],
            'sst2': [50.9],
            'stsb': [0, 0],
            'mrpc':[68.4, 81.2],
            'qqp': [63.2, 53.8],
            'mnli': [35.4],
            'qnli': [50.5],
            'rte': [52.7],
            'wnli': [56.3]
        }

        fig, axs = plt.subplots(3, 3, constrained_layout=True, figsize=(11, 6), dpi=resolution)

        for i, t in enumerate(tasks):
            task = t['task_name']
            axs[i//3, i%3].set_title(task)
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
                axs[i//3, i%3].plot([e*10 for e in epochs] if detailed else epochs, score, label=metrics[k], color=f'C{k}')
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
        plt.savefig(f'Plots/{model_name}/overview_{model_name}.png')
        plt.close()

def create_overview_detailed(models, tasks, add_gpt2=False, select_epochs='all'):
    
    freezed = False
            
    if add_gpt2:
        model_names = [m['name'] for m in models]
        if "small_model" in model_names:
            models.append({"name":"gpt2_small"})
        if "large_model" in model_names:
            models.append({"name":"gpt2_medium"})

    print("--------------Collecting data, this may take a while--------------")
    total_data={}
    for m in models:
        model = m['name']
        # creates a data structure holding all necessary values
        total_data[model] = {}
        if select_epochs=='custom':
            if not 'finetuning_epochs' in m:
                print("!!ERROR!!\nepoch selection strategy 'custom' requires the key 'finetuning_epochs' in the 'models' dictionary")
                return
            epochs=m['finetuning_epochs']
        elif select_epochs=='all':
            epochs=[(int(i/10) if isinstance(i/10, float) and (i/10).is_integer() else i/10) for i in range(2*10+1)]
        elif select_epochs=='last':
            epochs = [2]
        else:
            print(f"!!ERROR!!\nepoch selection strategy '{select_epochs}' is unknown.\n Choose one of 'custom', 'all', 'last'")
            return
        
        for t in tasks:
            task = t['task_name']
            task_res = {}
            for epoch in epochs:
                if 'gpt2' in model and epoch !=1:
                    continue
                sd = torch.load(f"FinetunedModels/{model}/({epoch}){model}/{'freezed_' if freezed else ''}{task}_({epoch}){model}.pt", map_location='cpu')['score_history']

                keys = sd[0].keys()
                scores = {k: [s[k] for s in sd] for k in keys}
                res = {k: sum(scores[k][-2:])/2 for k in keys}
                task_res[epoch] = res
            total_data[model][task] = task_res

    print("--------------Plotting--------------")
    for m in models:
        model_name = m['name']
        # Do not create an extra plot for gpt2 models
        if "gpt2" in model_name:
            continue
        data = total_data[model_name]

        baselines = {
            'cola': [0],
            'sst2': [50.9],
            'stsb': [0, 0],
            'mrpc':[68.4, 81.2],
            'qqp': [63.2, 53.8],
            'mnli': [35.4],
            'qnli': [50.5],
            'rte': [52.7],
            'wnli': [56.3]
        }

        fig, axs = plt.subplots(3, 3, constrained_layout=True, figsize=(11, 6), dpi=resolution)

        for i, t in enumerate(tasks):
            task = t['task_name']
            axs[i//3, i%3].set_title(task)
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
                axs[i//3, i%3].plot([e*10 for e in epochs], score, label=metrics[k], color=f'C{k}')
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
        plt.savefig(f'Plots/{model_name}/detailed_overview_{model_name}.png')
        plt.close()