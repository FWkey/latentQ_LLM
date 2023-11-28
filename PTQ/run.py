
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import quant

from gptq import GPTQ, Observer
from utils import find_layers, DEV, set_seed, get_wikitext2, get_ptb, get_c4, get_ptb_new, get_c4_new, get_loaders, export_quant_table, gen_conditions
from texttable import Texttable

from llama import get_llama,llama_sequential,llama_eval,llama_benchmark,llama_tune_act
# from bloom import get_bloom,bloom_sequential,bloom_eval,bloom_benchmark,bloom_tune_act
from opt import get_opt,opt_sequential,opt_eval #,opt_benchmark,opt_tune_act

# TODO: perform packing on GPU
def model_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    quant.make_quant_linear(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [quant.QuantLinear1,quant.QuantLinear0,quant.QuantLinear2])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        if isinstance(qlayers[name],quant.QuantLinear1):
            qlayers[name].config_act_func(args.act_fun)
        qlayers[name].pack(layers[name], scale, zero, g_idx)
        if hasattr(qlayers[name],'tmp_bias'):
            print(qlayers[name].inp_absmax.data,qlayers[name].input_bias.data)
            qlayers[name].inp_absmax.data = qlayers[name].input_bias.data+0.2
            qlayers[name].config_act_func(args.act_fun)
            if hasattr(qlayers[name], 'bias') and qlayers[name].bias is not None:
                qlayers[name].bias += qlayers[name].tmp_bias
            else: 
                if qlayers[name].bias is None:
                    delattr(qlayers[name],'bias')
                qlayers[name].register_buffer('bias',qlayers[name].tmp_bias)
    print('Done.')
    return model


def load_quant(model, checkpoint, wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True):
    from transformers import AutoConfig,modeling_utils,AutoModelForCausalLM

    model_config = AutoConfig.from_pretrained(model)
    
    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(model_config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    names = list(layers.keys())
    for name in names:
        if 'layer' not in name and 'h' not in name:
            del layers[name]
    quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        state_dict = model.state_dict()
        ckt = safe_load(checkpoint)
        for key in ckt.keys():
            try:
                state_dict[key].copy_(ckt[key])
            except Exception as e:
                print(key)
                print(e)
                pars = key.split('.')
                att = pars[-1]
                modname = '.'.join(pars[1:-1])
                for name,mod in model.named_modules():
                    if modname in name:
                        delattr(mod,att)
                        mod.register_buffer(att, ckt[key])


        # model.load_state_dict(ckt)
    else:
        model.load_state_dict(torch.load(checkpoint))

    for sublayer in model.modules():
        if isinstance(sublayer,quant.QuantLinear1):
            sublayer.config_act_func(args.act_fun)

    if not args.kvquant:
        for sublayer in model.modules():
            if hasattr(sublayer,'out_scale'):
                sublayer.out_scale *= 0       

    # quant.make_quant_attn(model)
    # if eval and fused_mlp:
    #     quant.make_fused_mlp(model)

    # if warmup_autotune:
    #     quant.autotune_warmup_linear(model, transpose=not (eval))
    #     if eval and fused_mlp:
    #         quant.autotune_warmup_fused(model)
    model.seqlen = 2048
    print('Done.')

    return model





if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='llama model to load')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'], help='Where to extract calibration data from.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--wbits', type=int, default=8, choices=[2, 3, 4, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')

    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--trits', action='store_true', help='Whether to use trits for quantization.')
    parser.add_argument('--test-generation', action='store_true', help='test generation.')
    parser.add_argument('--layers-dist', type=str, default='', help='Distribution of layers across GPUs. e.g. 2:1:1 for 2 layers on GPU 0, 1 layer on GPU 1, and 1 layer on GPU 2. Any remaining layers will be assigned to your last GPU.')

    parser.add_argument('--save', type=str, default='', help='Save quantized checkpoint under this name.')
    parser.add_argument('--save_safetensors', type=str, default='', help='Save quantized `.safetensors` checkpoint under this name.')
    parser.add_argument('--load', type=str, default='', help='Load quantized model.')

    parser.add_argument('--eval', action='store_true', help='evaluate quantized model.')
    parser.add_argument('--benchmark', type=int, default=0, help='Number of tokens to use for benchmarking.')
    parser.add_argument('--check', action='store_true', help='Whether to compute perplexity during benchmarking for verification.')

    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
    parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')
    parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')
    parser.add_argument('--new-eval', action='store_true', help='Whether to use the new PTB and C4 eval')
    parser.add_argument('--act_fun', type=str, default='per_tensor', help='activation quantization.')
    parser.add_argument('--smoothquant', action='store_true', help='whether to   ')
    parser.add_argument('--kvquant', action='store_true', help='whether to   ')
    parser.add_argument('--nogptq', action='store_true', help='whether to   ')
    parser.add_argument('--meanact', action='store_true', help='whether to   ')    
    parser.add_argument('--observe', action='store_true', help='whether to   ') 
    parser.add_argument('--nearest', action='store_true', help='whether to   ')
    args = parser.parse_args()

    if 'llama' in args.model:
        get_model = get_llama
        model_gptq = llama_sequential
        model_eval = llama_eval
        benchmark = llama_benchmark
        model_tune_act = llama_tune_act
    elif 'opt' in args.model:
        get_model = get_opt
        model_gptq = opt_sequential
        model_eval = opt_eval
    else:
        assert False

    if args.nogptq:
        quant.config['method'] = 'no'

    if args.layers_dist:
        gpu_dist = [int(x) for x in args.layers_dist.split(':')]
    else:
        gpu_dist = []

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize)
    else:
        model = get_model(args.model)
        model.eval()


    if not args.load and args.wbits < 16 :
        dataloader, testloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen)
        tick = time.time()
        quantizers = model_gptq(model, dataloader, DEV, local_args=args)
        model_pack(model, quantizers, args.wbits, args.groupsize)
        # model_tune_act(model, dataloader, DEV, local_args=args)
        print(time.time() - tick)

    if args.benchmark:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            model = model.to('cuda:0')
        else:
            model = model.to(DEV)
        dataloader, testloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen)
        input_ids = next(iter(dataloader))[0][:, :args.benchmark]
        benchmark(model, input_ids, check=args.check)

    if not args.observe and args.save:
        model.cpu()
        torch.save(model.state_dict(), args.save)

    if not args.observe and args.save_safetensors:
        model.cpu()
        from safetensors.torch import save_file as safe_save
        state_dict = model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        safe_save(state_dict, args.save_safetensors)

    if args.benchmark and not args.load:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            model = model.to('cuda:0')
        else:
            model = model.to(DEV)
        dataloader, testloader = get_loaders(args.dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
        input_ids = next(iter(dataloader))[0][:, :args.benchmark]
        benchmark(model, input_ids, check=args.check)
        print(args.dataset)

    if args.eval:
        datasets = ['wikitext2', 'ptb', 'c4']
        model.to(torch.bfloat16) 
        for dataset in datasets:
            dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
            print(dataset)
            model_eval(model, testloader, DEV, local_args=args)


