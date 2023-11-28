#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

import time
import deepspeed

# sys.path.append(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model
from utils.quant import MyAdam

DEBUG = False
def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='6,2,2',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `6,2,2`'
                        'will use 60% of data for phase 1, 20% for phase 2'
                        'and 20% for phase 3.')
    parser.add_argument(
        '--sft_only_data_path',
        nargs='*',
        default=[],
        help='Path to the dataset for only using in SFT phase.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--load_param",
        type=str,
        default='',
        help=
        "load quantized W4/A8W8 quantized model."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    parser.add_argument('--lbit',
                        type=int,
                        default=4,
                        help='4 for out method, or 16 for bf16 or 32 for fp32.')
    parser.add_argument('--obit',
                        type=int,
                        default=8,
                        help='8 for uint8, or 16 for float.')
    parser.add_argument('--port',
                        type=int,
                        default=1234)
    parser.add_argument('--gbit',
                        type=int,
                        default=16,
                        help='8 for uint8, or 16 for float.')
    parser.add_argument('--quant_schema',
                        type=str,
                        default='a8w8',
                        help='a8w8: per tensor 8bit activation + pre outchannel 8bit weight; a16w4: pre group (128) pre outchannel 4bit; qat: a8w8 with qat pipeline.')
    parser.add_argument('--lastlayer',
                        type=int,
                        default=0,
                        help='0 for all, -1 for except lmhead, n for only last n decoderlayers.')
    parser.add_argument('--qlora_ds',
                        type=str,
                        default='',
                        help='using qlora dataloader')
    parser.add_argument('--mmlu_eval',
                        type=str,
                        default='',
                        help='using mmlu evaluator.') 
    parser.add_argument('--mu_f',
                        type=int,
                        default=100,
                        help='mu_updata_f')
    parser.add_argument('--gradcheck',
                        action='store_true',
                        help='enable grad checkpoint')
    parser.add_argument('--offload',
                        action='store_true',
                        help='cpu offload for optimizer')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args

def dist_setup(rank, world_size, port='34877'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    os.environ['LOCAL_RANK'] = '0'
    torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)


import warnings
from utils.quant import QLinearA16W4,QLinearA8W8,QLinearQAT
from transformers.pytorch_utils import Conv1D
def main():
    args = parse_args()
    try:
        args.global_rank = torch.distributed.get_rank()
    except:
        dist_setup(0,1,str(args.port))
        args.global_rank = torch.distributed.get_rank()
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=False,
                                    stage=args.zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    # assert not args.offload, "zero-offload is not currently supported but coming soon!"
    #,tokenizer_type='llama' if 'llama' in args.model_name_or_path else None
    torch.distributed.barrier()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,use_fast = False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,use_fast = True)

    if 'llama' in args.model_name_or_path:
        tokenizer.eos_token = "</s>"
        tokenizer.eos_token_id = 2  # OPT eos-token-id

    # add pad token if not present
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config,
                            disable_dropout=False)

    tokenizer.abcd_idx = [
        tokenizer("A", add_special_tokens=False).input_ids[0],
        tokenizer("B", add_special_tokens=False).input_ids[0],
        tokenizer("C", add_special_tokens=False).input_ids[0],
        tokenizer("D", add_special_tokens=False).input_ids[0],
        ]
    
    num_layers = model.config.num_hidden_layers

    args.bf16_enable = True 

    decoder_name = ''
    for name, child in model.named_modules():
            if isinstance(child, torch.nn.ModuleList) and len(child) == num_layers:
                decoder_layers = child
                decoder_name = name + '.'
                break
    if not decoder_name:
        assert False, 'decoder not found!'
            
    if args.load_param and os.path.exists(args.load_param):
        checkpoint = args.load_param
        print('Loading model ...')
        if checkpoint.endswith('.safetensors'):
            from safetensors.torch import load_file as safe_load
            ckt = safe_load(checkpoint) 
        else:
            ckt = torch.load(checkpoint)
    else:
        assert False, 'load_param not exist!'

    if args.lastlayer > 0:
        print(f'NOTE: total decoder layers {num_layers}, quantize last {args.lastlayer} layers')  
        train_layers = decoder_layers[-args.lastlayer:] 
    else:
        print(f'NOTE: total decoder layers {num_layers}, quantize all layers')
        train_layers = decoder_layers

    i = -1    
    for module in list(decoder_layers):
      i += 1 
      wbit=4 if 'w4' in args.quant_schema else 8
      qat = 'qat' in args.quant_schema 
      for subname, submodule in list(module.named_modules()):
        for name, child in submodule.named_children():
            if isinstance(child, torch.nn.Linear) or isinstance(child, Conv1D):
                if isinstance(child, Conv1D):
                    child.weight.data = child.weight.transpose(0,1)
                if child.weight is None:
                    warnings.warn(f'{child} has empty weight!')
                    continue
                if wbit == 4:
                    newlayer = QLinearA16W4(child.weight.shape[1], child.weight.shape[0],bias=child.bias is not None,lbit=args.lbit,gbit=args.gbit)
                    tsubname = subname+'.' if subname else subname
                    newlayer.lname = decoder_name+str(i)+'.'+tsubname+name
                    newlayer.zeros.data = ckt[decoder_name+str(i)+'.'+tsubname+name+'.zeros'].to(torch.bfloat16)
                    newlayer.scales.data = ckt[decoder_name+str(i)+'.'+tsubname+name+'.scales'].to(torch.bfloat16)
                    newlayer.weight.set_value(ckt[decoder_name+str(i)+'.'+tsubname+name+'.qweight'], [newlayer.scales,newlayer.zeros,newlayer.wshape])
                    if args.lbit == 4:
                        newlayer.weight.residual.name = decoder_name+str(i)+'.'+tsubname+name+'.residual'
                elif qat and module in train_layers:
                    newlayer = QLinearQAT(child.weight.shape[1], child.weight.shape[0],bias=child.bias is not None,lbit=args.lbit,gbit=args.gbit)
                    tsubname = subname+'.' if subname else subname
                    newlayer.lname = decoder_name+str(i)+'.'+tsubname+name
                    newlayer.clampv.data = ckt[decoder_name+str(i)+'.'+tsubname+name+'.inp_absmax'].max()
                    newlayer.qscales.data = ckt[decoder_name+str(i)+'.'+tsubname+name+'.scales'].to(torch.bfloat16)
                    newlayer.weight.data = ((ckt[decoder_name+str(i)+'.'+tsubname+name+'.qweight'].to(torch.int32)-128)*newlayer.qscales).to(torch.bfloat16)
                else:
                    newlayer = QLinearA8W8(child.weight.shape[1], child.weight.shape[0],bias=child.bias is not None,lbit=args.lbit,gbit=args.gbit)
                    tsubname = subname+'.' if subname else subname
                    newlayer.lname = decoder_name+str(i)+'.'+tsubname+name
                    newlayer.clampv.data = ckt[decoder_name+str(i)+'.'+tsubname+name+'.inp_absmax'].max()
                    newlayer.qscales.data = ckt[decoder_name+str(i)+'.'+tsubname+name+'.scales'].to(torch.bfloat16)
                    ii = (ckt[decoder_name+str(i)+'.'+tsubname+name+'.qweight'].to(torch.int32)-128)
                    aa = (ii*newlayer.qscales).to(torch.bfloat16)
                    newlayer.weight.set_value(ii.to(torch.int8), newlayer.qscales)
                    if args.lbit == 4:
                        newlayer.weight.residual.name = decoder_name+str(i)+'.'+tsubname+name+'.residual'
                    else:
                        newlayer.weight.residual.data = aa
                if child.bias is not None:
                    newlayer.bias.data = ckt[decoder_name+str(i)+'.'+tsubname+name+'.bias'].to(torch.bfloat16)
                setattr(submodule, name, newlayer)
            elif hasattr(child,'weight') and child.weight is not None:
                tsubname = subname+'.' if subname else subname
                try:
                    child.weight.data = ckt[decoder_name+str(i)+'.'+tsubname+name+'.weight']
                except:
                    assert False, f"{decoder_name+str(i)+'+'+tsubname+name} not found!"
                if hasattr(child,'bias') and child.bias is not None:
                    child.bias.data = ckt[decoder_name+str(i)+'.'+tsubname+name+'.bias']


    if False:
        lm_head = model.lm_head
        wbitlm = 8
        newlayer = QLinear(lm_head.weight.shape[1], lm_head.weight.shape[0],bias=lm_head.bias is not None,lbit=args.lbit,wbit=wbitlm,gbit=args.gbit)
        newlayer.quant(lm_head.weight.contiguous(),dtype=torch.bfloat16 if args.bf16_enable else torch.float32)
        newlayer.add_hook()
        if lm_head.bias is not None:
            newlayer.bias.data = lm_head.bias.clone()
        setattr(model, 'lm_head', newlayer)


    train_phase = 1


    if args.qlora_ds:
        from qlora_utils import make_data_module, DataCollatorForCausalLM
        data_module = make_data_module(tokenizer=tokenizer, datasetname=args.qlora_ds)
        train_dataset, eval_dataset = data_module['train_dataset'],data_module['eval_dataset']
        data_collator = data_module['data_collator']
    else:
        train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        train_phase,
        args.seed,
        tokenizer,
        args.max_seq_len,
        sft_only_data_path=args.sft_only_data_path)
        data_collator = default_data_collator
    #     from qlora_utils import QLORA_train_evel
    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)

    train_dataloader = DataLoader(train_dataset,
                                collate_fn=data_collator,
                                sampler=train_sampler,
                                batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                collate_fn=data_collator,
                                sampler=eval_sampler,
                                batch_size=args.per_device_eval_batch_size)

    if args.mmlu_eval:
        from qlora_utils import getmmlu_dataset,mmlu_eval,DataCollatorForCausalLM
        mmlu_ds,mmlu_ds_test  = getmmlu_dataset(args.mmlu_eval)
        data_collator = DataCollatorForCausalLM(
            tokenizer=tokenizer,
            source_max_len=2048,
            target_max_len=512,
            train_on_source=False,
            predict_with_generate=False
        )
        mmlu_dataloader = DataLoader(mmlu_ds,
                                    collate_fn=data_collator,
                                    sampler=SequentialSampler(mmlu_ds),
                                    batch_size=2)
        mmlu_dataloader_test = DataLoader(mmlu_ds_test,
                                    collate_fn=data_collator,
                                    sampler=SequentialSampler(mmlu_ds_test),
                                    batch_size=2)

    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        step = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay)

    if args.lastlayer == 0:
        print('NOTE: train lm head!')
        model.lm_head.weight.requires_grad = True
        optimizer_grouped_parameters[0]["params"].append(model.lm_head.weight)


    for module in list(train_layers):
        for child in module.modules():
            if hasattr(child,'bias') and child.bias is not None:
                optimizer_grouped_parameters[1]["params"].append(child.bias)
            if isinstance(child, QLinearQAT) or isinstance(child, QLinearA8W8) or isinstance(child, QLinearA16W4) :
                child.weight.requires_grad = True
                child.add_hook()
                optimizer_grouped_parameters[0]["params"].append(child.weight)
            else:
                if hasattr(child,'weight') and child.weight.requires_grad:
                    optimizer_grouped_parameters[1]["params"].append(child.weight)


    print(0,len(optimizer_grouped_parameters[0]["params"]))
    print(1,len(optimizer_grouped_parameters[1]["params"]))

    optimizer = MyAdam(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            betas=(0.9, 0.95),obit=args.obit,mu_updata_f=args.mu_f,cpu=args.offload)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    if args.gradcheck:
        model.gradient_checkpointing_enable()
    #todo: argument in command : fp16 with loss scaling, fp32, bf16
    if args.deepspeed:
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            config=ds_config,
            lr_scheduler=lr_scheduler,
            dist_init_required=True)
        if args.bf16_enable:
            model.bfloat16()        
    else:
        if args.bf16_enable:
            model.bfloat16()
        model.to(device)    
    
    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, QLinearQAT) or isinstance(child, QLinearA8W8) or isinstance(child, QLinearA16W4):
                child.to(device)
                # if args.bf16_enable:
                #     child.totype(torch.bfloat16)


    # Train!
    model.eval()
    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    if DEBUG:
        perplexity = evaluation(model, eval_dataloader)
        print_rank_0(f"Initial ppl: {perplexity}", args.global_rank)
    if args.mmlu_eval:
        result = mmlu_eval(model,tokenizer,mmlu_dataloader,mmlu_ds)
        print_rank_0(result)
        mmlu_acc_val = result['accuracy']

    accum_num = args.gradient_accumulation_steps

    start = time.time()
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            if args.deepspeed:
                    model.backward(loss)
                    model.step()
            else:
                loss.backward()
                if step % accum_num == accum_num-1:
                    optimizer.step()
                    lr_scheduler.step()
                    model.zero_grad()
            torch.cuda.empty_cache()
            if (step % 20==10 and step < 1000) or step % 500 == 300:
                print(loss)
            if step % 1000 == 300:
                perplexity = evaluation(model, eval_dataloader)
                print_rank_0(f"Training ppl: {perplexity}", args.global_rank)
            #if step % 2000 == 1100:
                end = time.time()
                print_rank_0(f"Training time: {end-start} s")
                #perplexity = evaluation(model, eval_dataloader)
                #print_rank_0(f"ppl: {perplexity}", args.global_rank)
                if args.mmlu_eval:
                    result = mmlu_eval(model,tokenizer,mmlu_dataloader,mmlu_ds)
                    print_rank_0(result)
                    if result['accuracy'] > mmlu_acc_val:
                        mmlu_acc_val = result['accuracy']
                        print('#######################################test mmlu accuracy: ')
                        result = mmlu_eval(model,tokenizer,mmlu_dataloader_test,mmlu_ds)
                        print_rank_0(result)                        
                if DEBUG:
                    break
                model.train()
        print(loss)
        # Evaluate perplexity on the validation set.
        print_rank_0(
            f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)
        
    end = time.time()
    print_rank_0(f"All time: {end-start} s")

    perplexity = evaluation(model, eval_dataloader)
    print_rank_0(f"After train ppl: {perplexity}", args.global_rank)
    if args.mmlu_eval:
        result = mmlu_eval(model,tokenizer,mmlu_dataloader,mmlu_ds)
        print_rank_0(result)
    model.tput_timer.update_epoch_count()

    if args.output_dir is not None:
        model.to('cpu')
        print_rank_0('saving the final model ...', args.global_rank)
        for module in list(model.modules()):
            for child in module.modules():
                if isinstance(child, QLinearQAT) or isinstance(child, QLinearA8W8) or isinstance(child, QLinearA16W4):
                    child.before_save()
        # get the time of present
        now = time.strftime("%Y%m%d_%H%M", time.localtime())
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_{now}.pth"))

        # if args.global_rank == 0:
        #     save_hf_format(model, tokenizer, args)




if __name__ == "__main__":
    main()
