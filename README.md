# Latent weight quantization For LLM fine-tuning

## Introduction

Integerized training cannot save memory because the high precision latent weight, we provide latent weight quantization to reduce memory without accuracy degradation.

## Installation
### Requirement
You need a NVIDIA Ampere GPU to support Bfloat16, tested on A100.
### Dependency
Install library dependencies within an Anaconda environment, 'torch-int-bf16' is required for a8w8 calculation with bf16 output.

```bash
git clone --recurse-submodules https://github.com/FWkey/torch-int-bf16.git
conda create -n int_gpt python=3.8
conda activate int_gpt
conda install -c anaconda gxx_linux-64=9
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
source environment.sh
bash build_cutlass.sh
python setup.py install
cd ..
pip install -r requirements.txt
```

## Demo

Llama2-7b model fine-tuned on [GPT4-LLM dataset](teknium/GPT4-LLM-Cleaned):

1, Download the fine-tuned model:
```
wget -c 
```
For users in China, you can use this link:
```
wget -c 
```

2, run the command to generate output.

```
python main.py /path/to/your/config/file.yaml
```


## Usage
1, Generate quantized model in PTQ directory.


2, fine-tune a A8W8 model:
```
python  main0.py  --model_name_or_path facebook/opt-1.3b    --gradient_accumulation_steps 8 --per_device_train_batch_size 4 --per_device_eval_batch_size 8     --deepspeed  --port 33442    --lbit 4 --quant_schema a8w8 --lastlayer -1 --obit 8  --load_param ./PTQ/opt1b3_a8w8.safetensors
```

fine-tune a A16W4 model:
```
python  main0.py  --model_name_or_path facebook/opt-1.3b    --gradient_accumulation_steps 8 --per_device_train_batch_size 4 --per_device_eval_batch_size 8     --deepspeed  --port 33442    --lbit 4 --quant_schema a16w4 --lastlayer -1 --obit 8  --load_param ./PTQ/opt1b3w4.safetensors
```
