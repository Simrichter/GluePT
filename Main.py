import pretrain
import finetune
import evaluate
import plots

gpu_num=1

def minimal_setup():
    models = [
        # {'name':"minimal_model", 'max_epochs': 1, "embedding_dimension":128, "num_heads":4, "num_blocks":4, "accumulation_steps":8},
        {'name':"full_test", 'max_epochs': 1, "embedding_dimension":128, "num_heads":4, "num_blocks":4},
    ]
    
    tasks = [
        {'task_name': 'cola', 'tries':1},
        {'task_name': 'stsb', 'tries':1, 'epochs':2},
        {'task_name': 'sst2', 'tries':1, 'epochs':2},
        {'task_name': 'wnli', 'epochs':2, 'dropout':0.2, 'eval_interval':100},
        {'task_name': 'rte', 'tries':1},
        {'task_name': 'qnli', 'epochs':2, 'tries':1},
        {'task_name': 'mrpc', 'tries':1},
        {'task_name': 'qqp', 'epochs':0.5, 'eval_interval':500, 'tries':1},
        {'task_name': 'mnli', 'epochs':0.5, 'eval_interval':500, 'tries':1},
    ]
    
    # Pretrain all models
    print("-----starting pretraining-----")
    # pretrain.pretrain(models, gpu_num, num_subsets=1)
    print("-----pretraining finished-----")
    # Plot pretraining loss curves
    # plots.plot_pretraining(models, y_min=3.3, y_max=6.5)

    # Finetune all models on all tasks
    print("-----starting finetuning-----")
    # finetune.finetune(models, tasks, gpu_num)
    print("-----finetuning finished-----")
    # Evaluate the finetuned models on the test sets and create a zip
    evaluate.evaluate(models, tasks, gpu_num, select_epochs='last', evaluate_detailed=False, zip_only=False, validation_only=True)

def full_setup():
    models = [
        # {'name':"large_model", 'max_epochs':15},
        # {'name':"small_model", 'max_epochs': 15, "accumulation_steps":8},
        {'name':"gpt2_small", 'max_epochs': 0},
        {'name':"gpt2_medium", 'max_epochs': 0},
    ]
    
    tasks = [
        # {'task_name': 'cola', 'tries':7},
        # {'task_name': 'stsb', 'tries':5},
        # {'task_name': 'sst2', 'tries':2},
        # {'task_name': 'wnli', 'epochs':6, 'dropout':0.2, 'eval_interval':100},
        # {'task_name': 'rte', 'tries':10},
        # {'task_name': 'qnli', 'epochs':2},
        # {'task_name': 'mrpc', 'tries':3},
        # {'task_name': 'qqp', 'epochs':1, 'eval_interval':500},
        {'task_name': 'mnli', 'epochs':1, 'eval_interval':500},
    ]
    
    # Pretrain all models
    print("-----starting pretraining-----")
    pretrain.pretrain(models, gpu_num, num_subsets=1)
    print("-----pretraining finished-----")
    # Plot pretraining loss curves
    plots.plot_pretraining(models)
    # Finetune all models on all tasks
    print("-----starting finetuning-----")
    finetune.finetune(models, tasks, gpu_num, select_epochs='all')
    print("-----finetuning finished-----")

    # Evaluate the finetuned models on the test sets and create a zip
    # evaluate.evaluate(models, tasks, gpu_num, select_epochs='all', evaluate_detailed=False, zip_only=False)
    # plots.create_overview(models, tasks, add_gpt2, plot_differences=True)
    
def small_setup(): 
    models = [
        {'name':"small_model", 'max_epochs': 1, "accumulation_steps":8},
        {'name':"gpt2_small", 'max_epochs': 0, 'type':'gpt2_small'},
    ]

    tasks = [
            {'task_name': 'cola', 'tries':1},
            {'task_name': 'stsb', 'tries':1, 'epochs':2},
            {'task_name': 'sst2', 'tries':1, 'epochs':2},
            {'task_name': 'wnli', 'epochs':2, 'dropout':0.2, 'eval_interval':100},
            {'task_name': 'rte', 'tries':1},
            {'task_name': 'qnli', 'epochs':2, 'tries':1},
            {'task_name': 'mrpc', 'tries':1},
            {'task_name': 'qqp', 'epochs':0.5, 'eval_interval':500, 'tries':1},
            {'task_name': 'mnli', 'epochs':0.5, 'eval_interval':500, 'tries':1},
    ]

    # Pretrain all models
    print("-----starting pretraining-----")
    pretrain.pretrain(models, gpu_num, num_subsets=1)
    print("-----pretraining finished-----")
    # Plot pretraining loss curves
    plots.plot_pretraining(models, y_min=3.3, y_max=6.5)

    # Finetune all models on all tasks
    print("-----starting finetuning-----")
    finetune.finetune(models, tasks, gpu_num, select_epochs='last')
    print("-----finetuning finished-----")

    # Evaluate the finetuned models on the test sets and create a zip
    print("-----starting evaluation-----")
    evaluate.evaluate(models, tasks, gpu_num, select_epochs='last', evaluate_detailed=False, zip_only=False)
    print("-----evaluation finished-----")
    print("-----plotting overview-----")
    plots.create_overview(models, tasks, select_epochs='all', detailed=False)
    

def quick_setup():
    models = [
        # {'name':"large_model", 'max_epochs': 1, 'evals_per_epoch':10,  'compile_model':False, 'accumulation_steps':16},
        {'name':"small_model", 'max_epochs': 1, 'evals_per_epoch':10, 'compile_model':False, 'accumulation_steps':8},
        {'name':"gpt2_small", 'max_epochs': 0, 'compile_model':False, 'accumulation_steps':8},
        {'name':"gpt2_medium", 'max_epochs': 0, 'compile_model':False, 'accumulation_steps':16},
    ]
    
    tasks = [
        {'task_name': 'cola', 'epochs':0.1},
        {'task_name': 'stsb', 'epochs':0.1},
        {'task_name': 'sst2', 'epochs':0.1},
        {'task_name': 'wnli', 'epochs':0.1, 'dropout':0.2, 'eval_interval':100},
        {'task_name': 'rte', 'epochs':0.1},
        {'task_name': 'qnli', 'epochs':0.1},
        {'task_name': 'mrpc', 'epochs':0.1},
        {'task_name': 'qqp', 'epochs':0.01, 'eval_interval':500},
        {'task_name': 'mnli', 'epochs':0.01, 'eval_interval':500},
    ]
    
    # Pretrain all models
    print("-----starting pretraining-----")
    pretrain.pretrain(models, gpu_num, num_subsets=1)
    print("-----pretraining finished-----")
    # Plot pretraining loss curves
    plots.plot_pretraining(models)
    # Finetune all models on all tasks
    print("-----starting finetuning-----")
    finetune.finetune(models, tasks, gpu_num, select_epochs='last')
    print("-----finetuning finished-----")

    # Evaluate the finetuned models on the test sets and create a zip
    evaluate.evaluate(models, tasks, gpu_num, select_epochs='last', evaluate_detailed=False, zip_only=False, validation_only=True)
    # plots.create_overview(models, tasks, add_gpt2, plot_differences=True)
    
# full_setup()
# minimal_setup()
# small_setup()
quick_setup()
