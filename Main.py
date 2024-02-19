import pretrain
import finetune
import evaluate
import plots

gpu_num=1

models = [
    # {'name':"large_model", 'max_epochs':15, 'type':'large_model'},
    {'name':"small_model", 'max_epochs': 3, 'finetuning_epochs':[2], 'kwargs':{"detailed":True, "accumulation_steps":8}}
    #{'name':"gpt2_small", 'max_epochs': 15, 'type':'gpt2_small'}
    # {'name':"gpt2_medium", 'max_epochs': 15, 'type':'gpt2_medium'}
    # {'name':"tiny_model", 'max_epochs': 1, 'kwargs':{"embedding_dimension":16, "num_heads":1, "num_blocks":1, "batch_size":256, "accumulation_steps":8, "detailed":True}}
]

# tasks = [
#     # {'task_name': 'cola', 'tries':7}
#     {'task_name': 'stsb', 'tries':5}
#     # ,{'task_name': 'sst2', 'tries':2} #, 'epochs':3
#     # {'task_name': 'wnli', 'epochs':6, 'dropout':0.2, 'eval_interval':100} #Achtung, war 0.5
#     # ,{'task_name': 'rte', 'tries':10}
#     # ,{'task_name': 'qnli', 'epochs':2}
#     # ,{'task_name': 'mrpc', 'tries':3}
#     # ,{'task_name': 'qqp', 'epochs':1, 'eval_interval':10000}
#     # ,{'task_name': 'mnli', 'epochs':1, 'eval_interval':10000}    
#         ]

tasks = [
        {'task_name': 'cola', 'tries':1}
        ,{'task_name': 'stsb', 'tries':1}
        ,{'task_name': 'sst2', 'tries':1} #, 'epochs':3
        ,{'task_name': 'wnli', 'epochs':3, 'dropout':0.2, 'eval_interval':100} #Achtung, war 0.5
        ,{'task_name': 'rte', 'tries':1}
        ,{'task_name': 'qnli', 'epochs':2, 'tries':1}
        ,{'task_name': 'mrpc', 'tries':1}
        ,{'task_name': 'qqp', 'epochs':0.5, 'eval_interval':10000, 'tries':1}
        ,{'task_name': 'mnli', 'epochs':0.5, 'eval_interval':10000, 'tries':1}
]

# Pretrain all models
print("-----starting pretraining-----")
pretrain.pretrain(models, gpu_num, num_subsets=1)
print("-----pretraining finished-----")
# Plot pretraining loss curves
plots.plot_pretraining(models)

# Finetune all models on all tasks
print("-----starting finetuning-----")
finetune.finetune(models, tasks, gpu_num, select_epochs='custom', finetune_detailed=False)
print("-----finetuning finished-----")
# Plot finetuning loss curves
plots.plot_finetuning(models, tasks)

# Evaluate the finetuned models on the test sets and create a zip
evaluate.evaluate(models, tasks, gpu_num, select_epochs='custom', evaluate_detailed=False)
plots.create_overview(models, tasks, detailed=False)
