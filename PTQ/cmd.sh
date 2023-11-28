# python run.py ../hf_cache/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16/ wikitext2 --wbits 8 --save_sa llama7b_a8w8.safetensors --eval --sym --act_fun static --smoothquant
# python run.py ../hf_cache/models--huggyllama--llama-13b/snapshots/bf57045473f207bb1de1ed035ace226f4d9f9bba/ wikitext2 --nsamples 12 --wbit 4 --save_sa llama13bw4.safetensors --groupsize 128 --eval --nogptq --smoothquant

python run.py facebook/opt-1.3b wikitext2 --wbits 8 --save_sa opt1b3_a8w8.safetensors --eval --sym --act_fun static --smoothquant
python run.py facebook/opt-1.3b wikitext2 --nsamples 12 --wbit 4 --save_sa opt1b3w4.safetensors --groupsize 128 --eval --nogptq --smoothquant

python run.py facebook/opt-6.7b wikitext2 --wbits 8 --save_sa opt6b7_a8w8.safetensors --eval --sym --act_fun static --smoothquant
python run.py facebook/opt-6.7b wikitext2 --nsamples 12 --wbit 4 --save_sa opt6b7w4.safetensors --groupsize 128 --eval --nogptq --smoothquant

python run.py facebook/opt-13b wikitext2 --wbits 8 --save_sa opt13b_a8w8.safetensors --eval --sym --act_fun static --smoothquant
python run.py facebook/opt-13b wikitext2 --nsamples 12 --wbit 4 --save_sa opt13bw4.safetensors --groupsize 128 --eval --nogptq --smoothquant


python run.py huggyllama/llama-7b/ wikitext2 --wbits 8 --save_sa llama7b_a8w8.safetensors --eval --sym --act_fun static --smoothquant
python run.py huggyllama/llama-7b/ wikitext2 --nsamples 12 --wbit 4 --save_sa llama7bw4.safetensors --groupsize 128 --eval --nogptq --smoothquant

python run.py huggyllama/llama-13b/ wikitext2 --wbits 8 --save_sa llama13b_a8w8.safetensors --eval --sym --act_fun static --smoothquant
python run.py huggyllama/llama-13b/ wikitext2 --nsamples 12 --wbit 4 --save_sa llama13bw4.safetensors --groupsize 128 --eval --nogptq --smoothquant