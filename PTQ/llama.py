import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import quant

from gptq import GPTQ, Observer
from utils import find_layers, DEV, set_seed, get_wikitext2, get_ptb, get_c4, get_ptb_new, get_c4_new, get_loaders, export_quant_table, gen_conditions
from texttable import Texttable


from quant import smooth_module, mean_bias
from transformers.models.llama.modeling_llama import LlamaRMSNorm

def get_llama(model):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model.seqlen = 2048
    return model


@torch.no_grad()
def llama_sequential(model, dataloader, dev, local_args=None):
    if local_args is None:
        global args
    else:
        args = local_args

    print('Starting ...')
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):

        print(f'Quantizing layer {i+1}/{len(layers)}..')

        layer = layers[i].to(dev)
        full = find_layers(layer)
        if args.true_sequential:
            sequential = [['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'], ['self_attn.o_proj'], ['mlp.up_proj', 'mlp.gate_proj'], ['mlp.down_proj']]
        else:
            sequential = [list(full.keys())]


        def sta_batch_qkv(self, inp, out):
            # Hessian H = 2 X XT + λ I
            hidden_dim = out.shape[-1]
            comming_max = torch.max(out.view(-1, hidden_dim).abs().detach(), dim=0)[0].float().cpu()
            if not hasattr(self,'qkv_absmax'):
                self.qkv_absmax = comming_max
            else:
                self.qkv_absmax = torch.min(self.qkv_absmax, comming_max)

        def sta_batch_minmax(self, inp, out):
            # Hessian H = 2 X XT + λ I
            hidden_dim = out.shape[-1]
            comming_max = torch.max(out.view(-1, hidden_dim).detach(), dim=0)[0]
            comming_min = torch.min(out.view(-1, hidden_dim).detach(), dim=0)[0]
            if not hasattr(self,'out_max'):
                self.out_max = comming_max
            else:
                self.out_max = torch.max(self.out_max,comming_max)
            if not hasattr(self,'out_min'):
                self.out_min = comming_min
            else:
                self.out_min = torch.min(self.out_min,comming_min)

        def sta_batch0(self, inp, out):
            # Hessian H = 2 X XT + λ I
            hidden_dim = out.shape[-1]
            comming_max = torch.max(out.view(-1, hidden_dim).abs().detach(), dim=0)[0].float().cpu()
            if not hasattr(self,'out_absmax'):
                self.out_absmax = comming_max
            else:
                self.out_absmax = torch.max(self.out_absmax,comming_max)

        def sta_batch1(self, inps, out):
            # Hessian H = 2 X XT + λ I
            inp = inps[0]
            hidden_dim = inp.shape[-1]
            comming_max = torch.max(inp.view(-1, hidden_dim).abs().detach(), dim=0)[0].float().cpu()
            if not hasattr(self,'inp_absmax'):
                self.inp_absmax = comming_max
            else:
                self.inp_absmax = torch.max(self.inp_absmax,comming_max)


        if hasattr(args,'meanact') and args.meanact:
            handles = []
            for mod in layer.modules():
                if isinstance(mod, nn.LayerNorm) or isinstance(mod, LlamaRMSNorm):
                    handles.append(mod.register_forward_hook(sta_batch_minmax))
            for j in range(args.nsamples):
                layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()
            mean_bias(layer)
            for mod in layer.modules():
                if hasattr(mod, 'out_max'):
                    delattr(mod, 'out_max')
                if hasattr(mod, 'out_min'):
                    delattr(mod, 'out_min')

        if hasattr(args,'smoothquant') and args.smoothquant:
            if args.wbits<8:
                smooth_module(layer,weight_smooth=True)
            if args.act_fun == 'static':
                handles = []
                for mod in layer.modules():
                    if isinstance(mod, nn.LayerNorm) or isinstance(mod, LlamaRMSNorm):
                        handles.append(mod.register_forward_hook(sta_batch0))
                handles.append(layer.mlp.down_proj.register_forward_hook(sta_batch1))
                handles.append(layer.self_attn.o_proj.register_forward_hook(sta_batch1))
                if hasattr(args,'kvquant') and args.kvquant:
                    handles.append(layer.self_attn.k_proj.register_forward_hook(sta_batch_qkv))
                    handles.append(layer.self_attn.v_proj.register_forward_hook(sta_batch_qkv))
                    handles.append(layer.self_attn.q_proj.register_forward_hook(sta_batch_qkv))
                for j in range(args.nsamples):
                    layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                for h in handles:
                    h.remove()
                smooth_module(layer)
                for mod in layer.modules():
                    if hasattr(mod, 'inp_absmax'):
                        delattr(mod, 'inp_absmax')
                    if hasattr(mod, 'out_absmax'):
                        delattr(mod, 'out_absmax')
                if hasattr(args,'kvquant') and args.kvquant:
                    layer.self_attn.v_proj.v_absmax = layer.self_attn.v_proj.qkv_absmax
                    layer.self_attn.q_proj.qk_absmax = layer.self_attn.q_proj.qkv_absmax
                    layer.self_attn.k_proj.qk_absmax = layer.self_attn.k_proj.qkv_absmax



        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name], observe=args.observe)
                gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)

            def add_batch(name):

                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                if hasattr(args,'nogptq')  and args.nogptq:
                    if args.wbits<8:
                        scale, zero, g_idx, error = gptq[name].searchquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
                    else:
                        scale, zero, g_idx, error = gptq[name].naivequant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
                else:
                    scale, zero, g_idx, error = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
                quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)

                gptq[name].free()


        # for j in range(args.nsamples):
        #     outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]



        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        # print('+------------------+--------------+------------+-----------+-------+')
        # print('\n')

    model.config.use_cache = use_cache

    return quantizers

@torch.no_grad()
def llama_tune_act(model, dataloader, dev, local_args=None):
    if local_args is None:
        global args
    else:
        args = local_args

    def _norm(hidden_states):
        # variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # hidden_states = hidden_states * torch.rsqrt(variance + 1e-6)
        return hidden_states

    print('Tuning act clamp value ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    outs0 = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    # attention_mask = attention_mask.repeat_interleave(args.nsamples,dim=0)
    position_ids = cache['position_ids']
    # position_ids = position_ids.repeat_interleave(args.nsamples,dim=0)
        

    for i in range(len(layers)):
        print('Layer ',i, ' ....')
        layer = layers[i].to(dev)
        qlinlayers = find_layers(layer,[quant.QuantLinear])
        for name in qlinlayers:
            qlinlayers[name].shutdown_actquant()
        sequential = [['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'], ['self_attn.o_proj'], ['mlp.up_proj', 'mlp.gate_proj'], ['mlp.down_proj']]
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        # outs = layer(inps, attention_mask=attention_mask, position_ids=position_ids)[0]
        comp_outs = _norm(outs)
        for names in sequential:
            print(names)
            subset = {n: qlinlayers[n] for n in names}
            bestscale = 1.0
            for name in subset:
                qlinlayers[name].turnon_static_actquant(bestscale)
            for j in range(args.nsamples):
                outs0[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
    
            # outs0 = layer(inps, attention_mask=attention_mask, position_ids=position_ids)[0]
            comp_outs0 = _norm(outs0)
            bestmse = (comp_outs0-comp_outs).to(torch.float32).view(args.nsamples,-1).norm(dim=1).max()
            print('{} {:.3f}; '.format(bestscale, bestmse.item()) ,end=';')
            testscales = [1.1,1.2,1.35,1.5,1.7]
            for upscale in testscales:
                for name in subset:
                    qlinlayers[name].modify_clampv(upscale)
                for j in range(args.nsamples):
                    outs0[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                # outs0 = layer(inps, attention_mask=attention_mask, position_ids=position_ids)[0]
                comp_outs0 = _norm(outs0)
                curmse = (comp_outs0-comp_outs).to(torch.float32).view(args.nsamples,-1).norm(dim=1).max()
                print('{} {:.3f}; '.format(upscale, curmse.item())  ,end=';')
                if curmse<bestmse:
                    bestscale = upscale
                    bestmse = curmse
                    # outs = outs0
                
            for name in subset:
                qlinlayers[name].set_clampv(bestscale)
                qlinlayers[name].shutdown_actquant()
            print('bestscale: ', bestscale)
        # for j in range(args.nsamples):
        #     outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    qlinlayers = find_layers(model,[quant.QuantLinear])
    for name in qlinlayers:
        qlinlayers[name].modify_clampv()



@torch.no_grad()
def llama_eval(model, testenc, dev, local_args=None):
    if local_args is None:
        global args
    else:
        args = local_args
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = quant.Quantizer()
                quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantizer.quantize(W).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


# TODO: perform packing on GPU
def llama_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    quant.make_quant_linear(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [quant.QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print('Done.')
    return model


def load_quant(model, checkpoint, wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True):
    from transformers import LlamaConfig, LlamaForCausalLM, modeling_utils
    config = LlamaConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)

    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)
    model.seqlen = 2048
    print('Done.')

    return model


def llama_multigpu(model, gpus, gpu_dist):
    model.model.embed_tokens = model.model.embed_tokens.to(gpus[0])
    if hasattr(model.model, 'norm') and model.model.norm:
        model.model.norm = model.model.norm.to(gpus[0])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[0])

    cache = {'mask': None, 'position_ids': None}

    class MoveModule(nn.Module):

        def __init__(self, module, invalidate_cache):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
            self.invalidate_cache=invalidate_cache

        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)

            if cache['mask'] is None or cache['mask'].device != self.dev or self.invalidate_cache:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']

            if cache['position_ids'] is None or cache['position_ids'].device != self.dev or self.invalidate_cache:
                cache['position_ids'] = kwargs['position_ids'].to(self.dev)
            kwargs['position_ids'] = cache['position_ids']
            
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.layers
    from math import ceil
    if not gpu_dist:
        pergpu = ceil(len(layers) / len(gpus))
        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(0 if i == 0 or i == len(layers) -1 else gpus[(i-1) // pergpu]), i==0)
    else:
        assert gpu_dist[0] >= 2, "At least two layers must be on GPU 0."
        assigned_gpus = [0] * (gpu_dist[0]-1)
        for i in range(1, len(gpu_dist)):
            assigned_gpus = assigned_gpus + [i] * gpu_dist[i]

        remaining_assignments = len(layers)-len(assigned_gpus) - 1
        if remaining_assignments > 0:
            assigned_gpus = assigned_gpus + [-1] * remaining_assignments

        assigned_gpus = assigned_gpus + [0]

        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(gpus[assigned_gpus[i]]), i==0)

    model.gpus = gpus


def llama_benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    torch.cuda.synchronize()

    cache = {'past': None}

    def clear_past(i):

        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None

        return tmp

    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    max_memory = 0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(input_ids[:, i:i + 1], past_key_values=cache['past'], attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1)))
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            if hasattr(model, 'gpus'):
                mem_allocated = sum(torch.cuda.memory_allocated(gpu) for gpu in model.gpus) / 1024 / 1024
            else:
                mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            max_memory = max(max_memory, mem_allocated)
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        print('Median:', np.median(times))
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())
            print('max memory(MiB):', max_memory)

