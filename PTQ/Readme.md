# LLM Quantization for A16W4 and A8W8 

## Usage
You can quantize opt or llama model from huggingfacce or local path. Taking 'facebook/opt-1.3b' as an example, A16W4 quantization: 
```bash
python run.py facebook/opt-1.3b wikitext2 --nsamples 12 --wbit 4 --save_sa opt1b3w4.safetensors --groupsize 128 --eval --nogptq --smoothquant
```
A8W8 quantization:
```bash
python run.py facebook/opt-1.3b wikitext2 --wbits 8 --save_sa opt1b3_a8w8.safetensors --eval --sym --act_fun static --smoothquant
```

## Reference 
paper:  [GPTQ](https://arxiv.org/pdf/2210.17323.pdf) and [AWQ](https://arxiv.org/abs/2306.00978)

Code:   [GPTQ-for-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa) 