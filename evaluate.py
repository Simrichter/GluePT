import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import Model.Gpt as Gpt
import Data
import os
import csv
import shutil
from pathlib import Path

num_workers = 4

# A dict used to translate our task names to the file names required for a submission to GLUE
file_names = {'cola': 'CoLA', 'stsb': 'STS-B', 'sst2': 'SST-2', 'wnli': 'WNLI', 'rte': 'RTE', 'qnli': 'QNLI',
              'mrpc': 'MRPC', 'qqp': 'QQP', 'mnli-m': 'MNLI-m', 'mnli-mm': 'MNLI-mm', 'ax':"AX"}

# This dict stores the required number of predictions. It is used to check if all test samples have been labeled before creating the zip
task_sizes = {'CoLA': 1063, 'STS-B': 1379, 'SST-2': 1821, 'WNLI': 146, 'RTE': 3000, 'QNLI': 5463, 'MRPC': 1725,
              'QQP': 390965, 'MNLI-m': 9796, 'MNLI-mm': 9847, 'AX': 1104}

# This function translates the numerical label assigned by the model into word labels, which are expected on some of the GLUE tasks
def translate_pred(x, task):
        if 'mnli' in task or task =="ax":
            return "entailment" if x == 0 else ("neutral" if x == 1 else "contradiction")
        if task in ["rte", "qnli"]:
            return "entailment" if x == 0 else "not_entailment"
        return x

def prepare_x(x, task):
        if task == "stsb":
            x2 = torch.cat((x[1], x[0]), dim=1)
            x = torch.cat((x[0], x[1]), dim=1)
            return x, x2
        elif task in ['wnli', 'rte', 'qnli', 'mrpc', 'qqp', 'mnli-m', 'mnli-mm', 'ax']:
            return torch.cat((x[0], x[1]), dim=1)
        else:
            return x

def evaluate_GLUE(model, epoch, device):
    # Loads the test datasets for the task
    # MNLI has two test datasets (matched and mismatched)
    task = model.task
    if task == 'mnli-m':
        test_data = Data.FinetuneData('mnli', 'test_matched')
    elif task == 'mnli-mm':
        test_data = Data.FinetuneData('mnli', 'test_mismatched')
    else:
        test_data = Data.FinetuneData(task, 'test')

    # Creates the dataloader for the test set
    test_loader = DataLoader(test_data, num_workers=num_workers, shuffle=False, pin_memory=True, batch_size=1)

    model.eval()

    # Iterates over all test samples and collects the assigned labels in a list
    preds = []
    for data in tqdm(test_loader, desc=f"model: {model.name}, task: {task}"):
        x = prepare_x(data[0], task)

        # For STS-B, the two input sentences are passed through the network in both possible orders, just like during finetuning
        if task == "stsb":
            out = model(x[0].to(device))
            out2 = model(x[1].to(device))

            out += out2
            del out2
            out = torch.clamp(out, min=0, max=5) # Even though STS-B is a regression task, it's outputs must lie in the interval [0,5]. Therefore we clamp all larger or smaller values

        # For all other tasks, an argmax is used for sampling a label.
        else:
            out = torch.argmax(model(x.to(device)))

        preds.append(translate_pred(out.item(), task))

    # After all predictions are made, they are saved in a tsv file.
    path = os.path.join('Predictions', f"{'freezed_'if model.freeze else ''}({epoch}){model.name}")
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/{file_names[task]}.tsv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t', lineterminator='\n')
        writer.writerow(['index', 'prediction'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


# This method is used to create a zip file ready to be uploaded to "https://gluebenchmark.com/submit"
def create_zip(model_name, freeze_model=False):
    needed_files = ['QQP.tsv', 'CoLA.tsv', 'RTE.tsv', 'MNLI-m.tsv', 'QNLI.tsv', 'MNLI-mm.tsv', 'AX.tsv', 'WNLI.tsv', 'STS-B.tsv', 'SST-2.tsv', 'MRPC.tsv']# os.listdir("Predictions/Example_submission_Glue")
    actual_files = os.listdir(f"Predictions/{'freezed_'if freeze_model else ''}{model_name}")
    for nf in needed_files:
        # Check if the file is present in the "Predictions" folder. If a file is missing, the GLUE website will reject the submission
        if nf not in actual_files:
            print(f"!!ERROR!!\nFile '{nf}' not found in folder 'Predictions'\nA submission to GLUE requires all files")
            return
        
        # Check if each file contains the required number of labels. If the number of predictions is not correct, the GLUE website will reject the submission
        with open(f"Predictions/{'freezed_'if freeze_model else ''}{model_name}/{nf}") as file:
            line_count = len(file.readlines()) - 1  # -1 because the header line should not be counted
        assert line_count == task_sizes[Path(nf).stem], f"Model {model_name}, Task {Path(nf).stem}: line_count {line_count} does not match necessary size: {task_sizes[Path(nf).stem]}"
    
    # Jupyter notebooks create additional checkpoints. As they are only temporary files and cause problems when zipping the folder, we delete them
    if os.path.exists(f"Predictions/{'freezed_'if freeze_model else ''}{model_name}/.ipynb_checkpoints"):
        shutil.rmtree(f"Predictions/{'freezed_'if freeze_model else ''}{model_name}/.ipynb_checkpoints")

    # Finally, the zip file is created and placed in the folder "Zip-Out"
    shutil.make_archive(f"Zip-Out/{'freezed_'if freeze_model else ''}{model_name}", "zip", f"Predictions/{'freezed_'if freeze_model else ''}{model_name}")


# # This function collects the validation scores of the last evaluation of the finetuning.
# # Assuming the validation and test sets to be similar, this gives a rough estimation of what the test scores might look like.
# # However, the results are too optimistic, since the models with the best validation score are chosen during finetuning.
# def estimate(model_name, epoch, task, freeze_model=False):
#     if 'mnli' in task:  # On MNLI, we evaluate only on the matched set during finetuning, because the mismatched set yields very similar results
#         task = 'mnli'
#     state = torch.load(f"FinetunedModels/{model_name}/({epoch}){model_name}/{'freezed_'if freeze_model else ''}{task}_({epoch}){model_name}.pt", map_location='cpu')
#     keys = state['score_history'][0].keys()
#     scores = {k: [s[k] for s in state['score_history']] for k in keys}
#     res = [scores[k][-1] for k in keys]
#     return res


def evaluate(models, tasks, gpu_num, select_epochs='last', evaluate_detailed=False, zip_only=False, validation_only=False):
    
    device = torch.device(f'cuda:{gpu_num}') if torch.cuda.is_available() and not validation_only else 'cpu'

    # At first, the task list needs a few modifications
    # If mnli is in the tasks list, it is replaced by mnli-m and mnli-mm
    mnli_entries = [t for t in tasks if t['task_name'] == "mnli"]
    if len(mnli_entries) >0:
        tasks = [t for t in tasks if t['task_name'] != "mnli"]
        for entry in mnli_entries:
            entry["task_name"]="mnli-m"
            tasks.append(entry)
            cop = entry.copy()
            cop["task_name"]="mnli-mm"
            tasks.append(cop)
    
    # The diagnostic task "ax" contains only a test set and is provided as an analysis tool by GLUE
    # The GLUE authors evaluate on it using the MNLI classifier. We follow this practice
    if not "ax" in [t for t in tasks if t['task_name']]:
        tasks.append({"task_name":"ax", "ax":True})
        
    val_data = {}

    for m in models:
        model_name=m['name']
        val_data[model_name]={}
        if select_epochs=='custom':
            if not 'finetuning_epochs' in m:
                print("!!ERROR!!\nepoch selection strategy 'custom' requires the key 'finetuning_epochs' in the 'models' dictionary")
                return
            epochs=m['finetuning_epochs']
        elif select_epochs=='all':
            epochs = [*range(m['max_epochs']+1)]
            if evaluate_detailed: # If detailed mode is active, add sub-epochs
                sub_epochs = [(int(i/10) if isinstance(i/10, float) and (i/10).is_integer() else i/10) for i in range(min(2, m['max_epochs'])*10+1) if i not in [0, 10, 20]]
                epochs = sorted(epochs+sub_epochs)
        elif select_epochs=='last':
            epochs = [m['max_epochs']]
        else:
            print(f"!!ERROR!!\nepoch selection strategy '{select_epochs}' is unknown.\n Choose one of 'custom', 'all', 'last'")
            return
        for e in epochs:
            val_data[model_name][e]={}
            freeze = any([t.get('freeze_model', False) for t in tasks])
            if not zip_only:
                print(f"Evaluating model ({e}){model_name}")
                for task in tasks:
                    model, state = Gpt.create_model(model_name, e, device, evaluate=True, **task)
                    if not task['task_name'] == "ax":
                        val_data[model_name][e][task['task_name']] = state["score_history"][-1]
                    if not validation_only:
                        evaluate_GLUE(model, e, device)
            if not validation_only:
                print(f"Creating zip...")
                create_zip(f"({e}){model_name}", freeze)
    
    for name, epoch in val_data.items():
        print(f"\n\n\n_______________________________\n{name}:\n_______________________________\n")
        # print(epoch)
        l =  list(list(epoch.values())[0].keys())
        for t in l:
            metrics = ", ".join(list(list(epoch.values())[0][t].keys()))
            print(f"{t} ({metrics})\n----------------------------------")
            for e, s in epoch.items():
                score_string = ', '.join([f"{(score*100):.2f}" for score in s[t].values()])
                print(f"{e}: {score_string}")
            print("")
            