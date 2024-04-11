import pretrain
import finetune
import evaluate
import plots

###########################
# This file provides premade setups for testing.
# All modules are designed to work on the same "models" and "tasks" lists, which configure the behaviour.
# However, every step can also be started manually.

# The provided setups are not guaranteed to work on every GPU.
# For example, a NVIDIA P100 does not support torch.compile(), which is why this feature has to be deactivated in the "models" setup (set 'compile_model':False)
# Also the GPU's VRAM is a limiting factor, while a NVIDIA A100 with 40GB can perform pretraining of our large model with only 4 accumulation steps,
# a P100 requires 16. For the small model, the P100 requires 8.

# To execute one of the setups, go to the end of this file, set "gpu_num" to select the desired GPU and uncomment one of the function calls
###########################

def minimal_setup():
    models = [
        {'name':"min_test", 'max_epochs': 1, "embedding_dimension":128, "num_heads":4, "num_blocks":4, 'accumulation_steps':1, 'compile_model':True},
    ]
    
    tasks = [
        {'task_name': 'cola', 'epochs':2},
        {'task_name': 'stsb', 'epochs':2},
        {'task_name': 'sst2', 'epochs':2},
        {'task_name': 'wnli', 'epochs':4, 'dropout':0.2, 'eval_interval':5},
        {'task_name': 'rte'},
        {'task_name': 'qnli', 'epochs':0.6},
        {'task_name': 'mrpc'},
        {'task_name': 'qqp', 'epochs':0.3, 'eval_interval':500},
        {'task_name': 'mnli', 'epochs':0.3, 'eval_interval':500},
    ]
    
    # Pretrain all models
    print("-----starting pretraining-----")
    pretrain.pretrain(models, gpu_num, num_subsets=1)
    print("-----pretraining finished-----")
    # Plot pretraining loss curves
    plots.plot_pretraining(models, y_min=5, y_max=10)

    # Finetune all models on all tasks
    print("-----starting finetuning-----")
    finetune.finetune(models, tasks, gpu_num)
    print("-----finetuning finished-----")

    print("-----starting evaluation-----")
    # Evaluate the finetuned models on the test sets and create a zip
    evaluate.evaluate(models, tasks, gpu_num, select_epochs='last', validation_only=True)
    print("-----evaluation finished-----")
    # As we do not expect the results to be actually submitted to GLUE, this setup only reports validation results.
    # If you want a zip file for a submission, set 'validation_only' to False

def full_setup():
    models = [
        {'name':"large_model", 'max_epochs':15},
        {'name':"small_model", 'max_epochs': 15},
        {'name':"gpt2_small", 'max_epochs': 0},
        {'name':"gpt2_medium", 'max_epochs': 0},
    ]
    
    tasks = [
        {'task_name': 'cola', 'tries':7},
        {'task_name': 'stsb', 'tries':5},
        {'task_name': 'sst2', 'tries':2},
        {'task_name': 'wnli', 'epochs':6, 'tries':10, 'dropout':0.2, 'eval_interval':5},
        {'task_name': 'rte', 'tries':10},
        {'task_name': 'qnli', 'epochs':2},
        {'task_name': 'mrpc', 'tries':3},
        {'task_name': 'qqp', 'epochs':1, 'eval_interval':500},
        {'task_name': 'mnli', 'epochs':1, 'eval_interval':500},
    ]
    
    # Pretrain all models
    print("-----starting pretraining-----")
    pretrain.pretrain(models, gpu_num)
    print("-----pretraining finished-----")
    # Plot pretraining loss curves
    plots.plot_pretraining(models)
    # Finetune all models on all tasks
    print("-----starting finetuning-----")
    finetune.finetune(models, tasks, gpu_num, select_epochs='all')
    print("-----finetuning finished-----")

    print("-----starting evaluation-----")
    # Evaluate the finetuned models on the test sets and create a zip
    evaluate.evaluate(models, tasks, gpu_num, select_epochs='all')
    print("-----evaluation finished-----")
    print("-----plotting overview-----")
    plots.create_overview(models, tasks, add_gpt2=True, select_epochs='all', plot_differences=True)
    
    
    
# This is a setup, which should produce relatively good results but is still much faster than the full setup.
# Our stabilization approach is not active ('tries':1) and the results might therefore fluctuate considerably on some tasks
# 
def small_setup(): 
    models = [
        {'name':"small_model", 'max_epochs': 1, "accumulation_steps":8, "compile_model":True},
    ]

    tasks = [
            {'task_name': 'cola'},
            {'task_name': 'stsb', 'epochs':2},
            {'task_name': 'sst2', 'epochs':2},
            {'task_name': 'wnli', 'epochs':4, 'dropout':0.2, 'eval_interval':5},
            {'task_name': 'rte'},
            {'task_name': 'qnli', 'epochs':1},
            {'task_name': 'mrpc'},
            {'task_name': 'qqp', 'epochs':0.5, 'eval_interval':500},
            {'task_name': 'mnli', 'epochs':0.5, 'eval_interval':500},
    ]

    # Pretrain all models
    print("-----starting pretraining-----")
    pretrain.pretrain(models, gpu_num, num_subsets=5)
    print("-----pretraining finished-----")
    # Plot pretraining loss curves
    plots.plot_pretraining(models, y_min=3.3, y_max=6.5)

    # Finetune all models on all tasks
    print("-----starting finetuning-----")
    finetune.finetune(models, tasks, gpu_num, select_epochs='all')
    print("-----finetuning finished-----")

    # Evaluate the finetuned models on the test sets and create a zip
    print("-----starting evaluation-----")
    evaluate.evaluate(models, tasks, gpu_num, select_epochs='all')
    print("-----evaluation finished-----")
    print("-----plotting overview-----")
    plots.create_overview(models, tasks, select_epochs='all')
    

gpu_num=0
    
# full_setup()
# minimal_setup()
# small_setup()
