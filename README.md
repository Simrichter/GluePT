## GluePT
This Project was created as part of my bachelor thesis at TU Dortmund.
For an in-depth explanation of Generative Pretrained Transformers regarding their architecture, use-cases and implementation, as well as the results my models achieved, have a look at my thesis.

The code resembles a complete pipeline for pretraining GPT models on subsets of the [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) dataset and finetuning them to the nine tasks of the [GLUE benchmark](gluebenchmark.com).
Evaluation, plotting and creating the zip files necessary for submissions to the benchmark is also possible.

## Getting started
If you want to start a pipeline, have a look at the file "Main.py".

It is used to configure the training setups, including the GPT model's sizes, the hyperparameters of pretraining and finetuning and optional additions like plotting, evaluation or the creation of a submitable zip-file.

After downloading the project, edit the "Main.py" file according to your requirements and execute it.

## GPU requirements
All computations were done on NVIDIA A100 40GB GPUs, but the code should work on most GPUs.
However, GPTs are typically large language models (LLM) and are therefore very memory-intensive by nature.
Therefore small consumer-grade graphics cards might not be suitable for training decent GPT models.

If you run into "out-of-memory" errors, try increasing the number of accumulation steps.
This will reduce memory consumption by using smaller training batches. It will not affect the training results, since the employed gradient accumulation will result in the same total batch size, but with more forward-passes.

Using smaller models can also reduce both memory-usage and compute-time, however using too small GPT models will result in poor performance on the benchmark.

## License
This Code is distributed under the MIT License. See `License.txt` for more information.
If you use any of my code, I would be happy to know what it is used for. So please let me know :)
