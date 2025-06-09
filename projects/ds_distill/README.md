# Dataset Distillation via task vectors

This project contains code to 1) train LoRA experts on a specific task and 2) attempt to distill the knowledge from the task vector back into language space via a distillation style objective. 

## 1. Training Experts

Use a command like 
```
python train_expert.py -c llama_3_3b  -k dataset=sordonia/flan-10k-flat finetune_task_name=super_glue_rte_1_0_2  subsample_dev=10 subsample_test=10 eval_every=0.5 save_every=500 total_steps=50 subsample_train=10 library_id=hf://pclucas14/llama-3b-collection expert_name=rte_10_samples
```

Some things to know : 
1. `-c llama_3_3b` points to the arguments in `configs/llama_3_3b.json`. You can create additional configs and reference them in the same way
2. all other keyword arguments as passed in after `-k`. These arguments overwrite whatever default value was provided in the code or in the previous config. 
3. You can use `subsample_{train/dev/test}` to artificially reduce the size of the dataset. `subsample_train=1` will train on a single datapoint for `totel_steps` number of steps
4. We built a tool to store and upload expert checkpoints to huggingface, called an `ExpertLibrary`. When running with args `library_id=hf://pclucas14/llama-3b-collection expert_name=rte_10_samples`, you will see your expert listed at `https://huggingface.co/pclucas14/llama-3b-collection` (Change `pclucas14` with your HF id.)


## 2. Distilling Experts

The code to distill the task vector back into embedding space using the gradient mathing approach is `train_distill.py`. If you run 
```
python train_distill.py -k library_id=hf://pclucas14/llama-3b-collection expert_name=rte_10_samples total_steps=100 eval_every=50
```

It will load the expert trained in the previous step, and distill it. 

You can also try to distill using another approach, which seeks to find points where the base model and the task adapter model "disagree on" the most. The code for this is in `delta_distil.py`. Similarly, you can execute 

```
python delta_distill.py -k library_id=hf://pclucas14/llama-3b-collection expert_name=rte_10_samples total_steps=100 eval_every=50
```