import torch.nn as nn
import torch

@torch.no_grad()
def python_compress(fdata,bit=4):
    assert bit==4
    int_data = fdata.view(-1,8//bit).to(torch.int8)
    int_data[:,0] =  (int_data[:,0] << 4) + int_data[:,1]
    return int_data[:,0].contiguous()


@torch.no_grad()
def python_decompress(int_data,bit=4):
    assert bit==4
    numel_h = int_data.shape[0]
    fdata = torch.empty((numel_h,2),device=int_data.device,dtype = torch.int8)
    fdata[:,0] = (int_data >> 4) % 16
    fdata[:,1] = (int_data) % 16
    return fdata

## optimizer quantization utilies
@torch.no_grad()
def Qfunc_8bit_opt(tensor, blocksize=1,  qparam = None):
    tensor_shape = tensor.shape
    tensor = tensor.view(blocksize,-1)
    if qparam is None:
        t_min, t_max = torch.aminmax(tensor,dim=1,keepdim =True)
        scale = (t_max-t_min) / 255 + 1e-10
    else: 
        scale,t_max = qparam
    qtensor = torch.clamp_(torch.round_( (t_max - tensor).div_(scale) ), 0, 255)
    qtensor = qtensor.to(torch.uint8).view(tensor_shape)
    return qtensor, [scale,t_max]

@torch.no_grad()
def DeQfunc_8bit_opt(qtensor, qparam, dtype = torch.float32) -> torch.Tensor:
    scale,t_max = qparam
    tensor_shape = qtensor.shape
    qtensor = qtensor.view(scale.shape[0],-1)
    tensor = t_max - qtensor.to(dtype)*(scale)
    return tensor.view(tensor_shape).to(dtype)

## A16W4 optimizer quantization utilies
GROUPSIZE = 128
@torch.no_grad()
def Qfunc_4bit_g_res(tensor,residual_handle,  qparam, groupsize = GROUPSIZE):
    scales,zeros,shape = qparam
    scales = scales.contiguous().reshape(-1, 1)
    zeros = zeros.contiguous().reshape(-1, 1)
    tensor = tensor.data.view(-1, groupsize)
    Z = tensor / scales+zeros
    qtensor = torch.round(Z)
    residual_fp = Z - qtensor
    if isinstance(residual_handle,ResidualTensor):
        adj = residual_handle.set_value(residual_fp.view(shape),scales)
        qtensor = qtensor.view(shape)
        qtensor += adj
    else:
        residual_handle.data = tensor.view(shape).to(residual_handle.dtype)

    intweight = torch.clamp(qtensor, 0, 15).to(torch.uint8)
    qweight = python_compress(intweight)
    return qweight


@torch.no_grad()
def DeQfunc_4bit_g(qtensor, qparam, dtype = torch.float32):
    scales,zeros,shape = qparam
    fintweight = python_decompress(qtensor).view(-1, GROUPSIZE)
    fweight = (fintweight - zeros)*scales
        
    return fweight.view(shape).to(dtype)


## A8W8 optimizer quantization utilies
# @torch.no_grad()
# def Qfunc_8bit(t, qscale):
#     # q_max = 2**(n_bits-1)-1
#     t.div_(qscale).round_().clamp_(-128,127).to(torch.int8)
#     return t

# @torch.no_grad()
# def DeQfunc_8bit(t, qscale):
#     # q_max = 2**(n_bits-1)-1
#     return t*qscale

#todo: fuse Qfunc_res with residual quantization into one cuda function
@torch.no_grad()
def Qfunc_8bit_res(tensor, residual_handle, qscales):
    tensor_shape = tensor.shape
    # tensor = tensor.view(blocksize,-1)
    qtensor = tensor/(qscales)
    qtensor.round_()
    if residual_handle is None:
        pass
    elif isinstance(residual_handle,ResidualTensor):
        residual_fp = tensor/(qscales) - qtensor
        adj = residual_handle.set_value(residual_fp.view(tensor_shape),qscales)
        qtensor += adj
    else:
        residual_handle.data = tensor.to(residual_handle.dtype)
    qtensor = qtensor.clamp_(-128,127).to(torch.int8)
    return qtensor




from torch_int._CUDA import linear_a8_w8_bfp32_ofp32,linear_a8_w8_bbf16_obf16
global dict_temp
dict_temp = []

class QLinearFunctionW4(torch.autograd.Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, weight_d, qparamw4, bias):
        weight_forward = DeQfunc_4bit_g(weight_d,qparamw4,input.dtype)
        output = input.mm(weight_forward.t())
        if bias is not None:
            output += bias #bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        input, weight_d, qparamw4, bias = inputs
        ctx.save_for_backward(input, weight_d.data, bias)
        ctx.save_qparamw4 = qparamw4

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight_d, bias = ctx.saved_tensors
        qparamw4 = ctx.save_qparamw4
        weight_forward = DeQfunc_4bit_g(weight_d,qparamw4,input.dtype)
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight_forward)
        grad_weight = grad_output.t().mm(input)
        dict_temp.append(grad_weight)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, None, None, grad_bias


class QLinearFunctionW4_eval(torch.autograd.Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, weight_d, qparamw4, bias):
        weight_forward = DeQfunc_4bit_g(weight_d,qparamw4,input.dtype)
        output = input.mm(weight_forward.t())
        if bias is not None:
            output += bias #bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        pass

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None

# Inherit from Function
class QLinearFunctionA8W8_static(torch.autograd.Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, weight_d, qparamaw, bias):
        clampv,qscales = qparamaw            
        input8 = input*127./clampv
        input8.round_().clamp_(-128,127)
        if bias is not None:
            output = linear_a8_w8_bbf16_obf16( input8.to(torch.int8), weight_d, bias/qscales.view(-1), clampv/127.)*qscales.view(1,-1)
        else:
            output = linear_a8_w8_bbf16_obf16( input8.to(torch.int8), weight_d, 0*qscales.view(-1), clampv/127.)*qscales.view(1,-1)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight_d, qparamaw, bias = inputs
        ctx.save_for_backward(input, weight_d.data, bias)
        ctx.save_qparamaw = qparamaw

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight_d, bias = ctx.saved_tensors
        clampv,qscales = ctx.save_qparamaw             
        int_s = 127./clampv
        input_f = input*int_s
        input_f.round_().clamp_(-128,127).div_(int_s)
        weight_forward = weight_d.to(input.dtype)
        weight_forward *= qscales
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight_forward)
        grad_weight = grad_output.t().mm(input_f)
        dict_temp.append(grad_weight)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, None, None, grad_bias

# Inherit from Function
class QLinearFunctionA8W8_dynamic0(torch.autograd.Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(ctx,input, weight_d, qparamaw, bias):
        clampv,qscales = qparamaw
        clampv = input.max().item() 
        ctx.save_for_backward(input, weight_d.data, bias)
        ctx.save_qparamaw = (clampv,qscales)

        input8 = input*127./clampv
        input8.round_().clamp_(-128,127)
        if bias is not None:
            output = linear_a8_w8_bbf16_obf16( input8.to(torch.int8), weight_d, bias/qscales.view(-1), clampv/127.)*qscales.view(1,-1)
        else:
            output = linear_a8_w8_bbf16_obf16( input8.to(torch.int8), weight_d, 0*qscales.view(-1), clampv/127.)*qscales.view(1,-1)
        return output


    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight_d, bias = ctx.saved_tensors
        clampv,qscales = ctx.save_qparamaw             
        int_s = 127./clampv
        input_f = input*int_s
        input_f.round_().clamp_(-128,127).div_(int_s)
        weight_forward = weight_d.to(input.dtype)
        weight_forward *= qscales
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight_forward)
        grad_weight = grad_output.t().mm(input_f)
        dict_temp.append(grad_weight.data)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, None, None, grad_bias
    
class QLinearFunctionA8W8_dynamic(torch.autograd.Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(ctx,input, weight_d, qparamaw, bias):
        clampv,qscales = qparamaw
        clampv = input.max().data
        # ctx.save_for_backward(input, weight_d.data, bias)
        ctx.save_qparamaw = (clampv,qscales)

        input8 = input*127./clampv
        input8.round_().clamp_(-128,127)
        input8 = input8.to(torch.int8)
        ctx.save_for_backward(input8, weight_d.data, bias)
        if bias is not None:
            output = linear_a8_w8_bbf16_obf16( input8, weight_d, bias/qscales.view(-1), clampv/127.)*qscales.view(1,-1)
        else:
            output = linear_a8_w8_bbf16_obf16( input8, weight_d, 0*qscales.view(-1), clampv/127.)*qscales.view(1,-1)
        return output


    @staticmethod
    def backward(ctx, grad_output):
        input8, weight_d, bias = ctx.saved_tensors
        clampv,qscales = ctx.save_qparamaw            
        int_s = clampv/127.
        input_f = input8.to(grad_output.dtype)*(int_s)
        weight_forward = weight_d
        weight_forward *= qscales
        weight_forward = weight_forward.to(grad_output.dtype)
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight_forward)
        grad_weight = grad_output.t().mm(input_f)
        dict_temp.append(grad_weight.data)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, None, None, grad_bias

# Inherit from Function
class QLinearFunctionA8W8_static_eval(torch.autograd.Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(ctx,input, weight_d, qparamaw, bias):
        clampv,qscales = qparamaw
        input8 = input*127./clampv
        input8.round_().clamp_(-128,127)
        if bias is not None:
            output = linear_a8_w8_bbf16_obf16( input8.to(torch.int8), weight_d, bias/qscales.view(-1), clampv/127.)*qscales.view(1,-1)
        else:
            output = linear_a8_w8_bbf16_obf16( input8.to(torch.int8), weight_d, 0*qscales.view(-1), clampv/127.)*qscales.view(1,-1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None
# Inherit from Function
class QLinearFunctionA8W8_dynamic_eval(torch.autograd.Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(ctx,input, weight_d, qparamaw, bias):
        clampv,qscales = qparamaw
        clampv = input.max().item() 
        input8 = input*127./clampv
        input8.round_().clamp_(-128,127)
        if bias is not None:
            output = linear_a8_w8_bbf16_obf16( input8.to(torch.int8), weight_d, bias/qscales.view(-1), clampv/127.)*qscales.view(1,-1)
        else:
            output = linear_a8_w8_bbf16_obf16( input8.to(torch.int8), weight_d, 0*qscales.view(-1), clampv/127.)*qscales.view(1,-1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None

# Inherit from Function
class QLinearFunctionQAT_dynamic(torch.autograd.Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight_f, qparamaw, bias):
        clampv,qscales = qparamaw
        clampv = input.max().item() 
        ctx.save_for_backward(input, weight_f.data, bias)
        ctx.save_qparamaw = (clampv,qscales)
        input8 = input*127./clampv
        input8.round_().clamp_(-128,127)
        weight_d = weight_f/(qscales)
        weight_d.round_().clamp_(-128,127)
        if bias is not None:
            output = linear_a8_w8_bbf16_obf16( input8.to(torch.int8), weight_d.to(torch.int8), bias/qscales.view(-1), clampv/127.)*qscales.view(1,-1)
        else:
            output = linear_a8_w8_bbf16_obf16( input8.to(torch.int8), weight_d.to(torch.int8), 0*qscales.view(-1), clampv/127.)*qscales.view(1,-1)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight_f, bias = ctx.saved_tensors
        clampv,qscales = ctx.save_qparamaw
        int_s = 127./clampv
        input_f = input*int_s
        input_f.round_().clamp_(-128,127).div_(int_s)
        
        weight_forward = weight_f/(qscales)
        weight_forward.round_().clamp_(-128,127).mul_(qscales)
        weight_forward = weight_forward.to(input.dtype)
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight_forward)
        grad_weight = grad_output.t().mm(input_f)
        dict_temp.append(grad_weight)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, None, None, grad_bias

class QLinearFunctionQAT_dynamic_eval(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight_f, qparamaw, bias):
        clampv,qscales = qparamaw
        clampv = input.max().item()
        input8 = input*127./clampv
        input8.round_().clamp_(-128,127)
        weight_d = weight_f/(qscales)
        weight_d.round_().clamp_(-128,127)
        if bias is not None:
            output = linear_a8_w8_bbf16_obf16( input8.to(torch.int8), weight_d.to(torch.int8), bias/qscales.view(-1), clampv/127.)*qscales.view(1,-1)
        else:
            output = linear_a8_w8_bbf16_obf16( input8.to(torch.int8), weight_d.to(torch.int8), 0*qscales.view(-1), clampv/127.)*qscales.view(1,-1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None

from typing import Union, Tuple, Optional
from torch.autograd import Variable
def backward_hook_fn(module, grad_out, grad_in):
    """ Output the grad of model output wrt. layer (only positive) """
    #todo: grad quantization
    assert len(dict_temp) == 1
    if isinstance(module.weight.grad,GradientQTensor):
        if module.weight.grad.acc_ite == 0:
            module.weight.grad.set_value(dict_temp.pop())
        else:
            module.weight.grad.add_(dict_temp.pop())    
    else:
        if module.weight.grad is None:
            module.weight.grad = dict_temp.pop()
        else:
            module.weight.grad.add_(dict_temp.pop())
    
class QLinearA16W4(nn.Module):
    def __init__(self, input_features, output_features, bias=True, lbit=0,gbit=16):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            self.register_parameter('bias', None)
        # save parameter state for inference
        self.groupsize = groupsize = GROUPSIZE
        # self.register_buffer('qweight', torch.zeros((input_features // 2*output_features), dtype=torch.int32))
        self.register_buffer('qweight', torch.tensor([]))
        self.register_buffer('zeros', torch.zeros((math.ceil(input_features / groupsize)*output_features,1), dtype=torch.bfloat16))
        self.register_buffer('scales', torch.zeros((math.ceil(input_features / groupsize)*output_features,1), dtype=torch.bfloat16))
        
        self.weight = QuantTensor([output_features, input_features],lbit=lbit,wbit=4,gbit=gbit) 

        self.wshape = [output_features, input_features]
        self.hook_handle = None

    def add_hook(self):
        self.hook_handle = self.register_full_backward_hook(backward_hook_fn)

    #     # Not a very smart way to initialize weights
    # def quant(self, weight_fp,dtype = torch.bfloat16):
    #     qweight, qparam = Qfunc_4bit_g_res(weight_fp,self.weight.residual,[self.scales,self.zeros,self.wshape],self.groupsize)
    #     self.weight.set_value(qweight, qparam)
        

    def before_save(self): # save parameter state for inference
        self.qweight.data = self.weight.data.cpu()

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.output_features, )
        if self.hook_handle is None:
            out = QLinearFunctionW4_eval.apply(x.reshape(-1, x.shape[-1]), self.weight.data, self.weight.qparam, self.bias)
        else:
            out = QLinearFunctionW4.apply(x.reshape(-1, x.shape[-1]), self.weight.data, self.weight.qparam, self.bias)
        
        return out.reshape(out_shape)
    
    def forward_eval(self, x):
        out_shape = x.shape[:-1] + (self.output_features, )
        out = QLinearFunctionW4_eval.apply(x.reshape(-1, x.shape[-1]), self.weight.data, self.weight.qparam, self.bias)
        
        return out.reshape(out_shape)

    def to(self, device: Optional[Union[int, torch.device]]):

        self.weight = self.weight.to(device=device)
        self.zeros = self.zeros.to(device=device)
        self.scales = self.scales.to(device=device)
        self.weight.set_value(None, [self.scales,self.zeros,self.wshape])
        if self.bias is not None:
            self.bias.to(device)

    # def totype(self, dtype):
    #     self.weight = self.weight.to(dtype=dtype)
    #     if self.bias is not None:
    #         self.bias.to(dtype)


    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'QLinearA16W4 input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )


class QLinearA8W8(nn.Module):
    def __init__(self, input_features, output_features, bias=True, lbit=0,gbit=8):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            self.register_parameter('bias', None)
        # save parameter state for inference
        self.register_buffer('qweight', torch.tensor([]))
        self.register_buffer('clampv', torch.ones(1))
        self.register_buffer('qscales', torch.zeros(output_features,1, dtype=torch.bfloat16))
        
        self.weight = QuantTensor([output_features, input_features],lbit=lbit,wbit=8,gbit=gbit) 

        self.wshape = [output_features, input_features]
        self.hook_handle = None

    def add_hook(self):
        self.hook_handle = self.register_full_backward_hook(backward_hook_fn)


    def before_save(self): # save parameter state for inference
        self.qweight.data = self.weight.data.cpu()

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.output_features, )
        if self.hook_handle is None:
            out = QLinearFunctionA8W8_dynamic_eval.apply(x.reshape(-1, x.shape[-1]), self.weight.data, (self.clampv.item(),self.qscales.data), self.bias)
        else:
            out = QLinearFunctionA8W8_dynamic.apply(x.reshape(-1, x.shape[-1]), self.weight.data, (self.clampv.item(),self.qscales.data), self.bias)
        
        return out.reshape(out_shape)
    
    def forward_eval(self, x):
        out_shape = x.shape[:-1] + (self.output_features, )
        out = QLinearFunctionA8W8_dynamic_eval.apply(x.reshape(-1, x.shape[-1]), self.weight.data, (self.clampv.item(),self.qscales.data), self.bias)
        
        return out.reshape(out_shape)

    def to(self, device: Optional[Union[int, torch.device]]):
        self.weight = self.weight.to(device=device)
        # self.clampv = self.clampv.to(device=device)
        self.qscales = self.qscales.to(device=device)
        self.weight.set_value(None, self.qscales)
        if self.bias is not None:
            self.bias.to(device)


    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'QLinearA8W8 input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )


class QLinearQAT(nn.Module):
    def __init__(self, input_features, output_features, bias=True, lbit=0,gbit=16):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            self.register_parameter('bias', None)
        # save parameter state for inference
        self.register_buffer('qweight', torch.tensor([]))
        self.register_buffer('clampv', torch.ones(1))
        self.register_buffer('qscales', torch.zeros(output_features,1, dtype=torch.bfloat16))

        assert gbit == 16
        self.weight = FPTensor(torch.empty((output_features, input_features), dtype=torch.bfloat16))
        self.wshape = [output_features, input_features]
        self.hook_handle = None

    def add_hook(self):
        self.hook_handle = self.register_full_backward_hook(backward_hook_fn)


    def before_save(self): # save parameter state for inference
        qtensor = self.weight.data/self.qscales
        qtensor.round_().clamp_(-128,127)
        self.qweight.data = qtensor.cpu().to(torch.int8)

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.output_features, )
        if self.hook_handle is None:
            out = QLinearFunctionQAT_dynamic_eval.apply(x.reshape(-1, x.shape[-1]), self.weight.data, (self.clampv.item(),self.qscales.data), self.bias)
        else:
            out = QLinearFunctionQAT_dynamic.apply(x.reshape(-1, x.shape[-1]), self.weight.data, (self.clampv.item(),self.qscales.data), self.bias)
        
        return out.reshape(out_shape)
    
    def forward_eval(self, x):
        out_shape = x.shape[:-1] + (self.output_features, )
        out = QLinearFunctionQAT_dynamic_eval.apply(x.reshape(-1, x.shape[-1]), self.weight.data, (self.clampv.item(),self.qscales.data), self.bias)
        
        return out.reshape(out_shape)

    def to(self, device: Optional[Union[int, torch.device]]):
        self.weight = self.weight.to(device=device)
        
        if self.bias is not None:
            self.bias.to(device)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'QLinearQAT input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

def prodx(ls):
    from functools import reduce
    return reduce(lambda x, y: x*y, ls)

#todo: support W4 quantization
class QuantTensor(object):
    def __init__(self, shape,wbit=8,gbit=8,lbit=4):
        self.shape = shape
        self.wbit = wbit
        self.initial = False
        self.qparam = None  
        self.lbit = lbit      
        if lbit==4:
            num_channels = shape[0] if wbit == 8 else prodx(self.shape)//GROUPSIZE
            qshape = [num_channels,prodx(self.shape)//num_channels]
            self.residual = ResidualTensor(qshape=qshape,oshape=shape)
        else:
            if lbit==16:
                self.residual = torch.empty(shape,dtype=torch.bfloat16)
            else:
                self.residual = torch.empty(shape,dtype=torch.float32)
        self.requires_grad = True
        if gbit == 16:
            self.grad = None
        else:
            self.grad = GradientQTensor(self.shape,bit=gbit)

    def set_value(self, data, qparam=None):        
        if data is not None:
            self.data = data.clone()
        self.qparam = qparam
        self.initial = True
        self.requires_grad = True

    def numel(self):
        return prodx(self.shape)
    
    def get_self(self):
        return self

    def __repr__(self):
        return "Quant value:\n{}\n with param:\n{}".format(self.data, self.qparam)
    
    def requires_grad_(self,boolv):
        self.requires_grad = boolv
        return self
    
    def to(self, device=None, dtype=None):
        if device is not None:
            self.data = self.data.to(device)
            if self.lbit > 8:
                self.residual = self.residual.to(device)
        return self


    @property
    def device(self):
        return self.data.device
    
    
    @property
    def type(self):
        return self.data.type


class FPTensor(object):
    def __init__(self, data, gbit=16):
        self.data = data
        self.shape = data.shape
        self.initial = False
        self.qparam = None  
        self.requires_grad = True
        if gbit == 16:
            self.grad = None
        else:
            self.grad = GradientQTensor(self.shape,bit=gbit)


    def numel(self):
        return prodx(self.shape)
    
    def get_self(self):
        return self

    def __repr__(self):
        return "Quant value:\n{}\n with param:\n{}".format(self.data, self.qparam)
    
    def requires_grad_(self,boolv):
        self.requires_grad = boolv
        return self
    
    def to(self, device=None, dtype=None):
        if device is not None:
            self.data = self.data.to(device)
        return self


    @property
    def device(self):
        return self.data.device
    
    
    @property
    def type(self):
        return self.data.type


def three_fold_curve_4bit(mu):
    if mu==3:
        return 2, 0.4330623539159437, 4, 0.46875
    elif mu==2:
        return 2, 0.37272211741724515, 4, 0.4375
    elif mu==1:
        return 2, 0.2666300266320331, 4, 0.375
    else:
        return 2, 0.125, 4, 0.25
    
from torch.utils.cpp_extension import load
cuda_module = load(name="MyQuantDequant", sources=["utils/cuda_src/multi_random_quant.cpp", "utils/cuda_src/multi_random_quant_kernel.cu"], verbose=True)
# import MyQuantDequant

EXT_type=torch.bfloat16
@torch.no_grad()
def cuda_quant_linear(residual_fp,q_tensor) : # torch.uint8
    #todo
    residual_fp=residual_fp.to(EXT_type)
    assert residual_fp.is_contiguous()
    if not residual_fp.is_cuda:
        return python_quant_linear(residual_fp) 
    # n_elements = residual_fp.numel()
    # q_tensor = torch.empty(((n_elements)//8,1),dtype=torch.int32,device=residual_fp.device) 
    cuda_module.uniform_quant(residual_fp,q_tensor)
    return residual_fp

@torch.no_grad()
def cuda_quant_3fold(residual_fp: torch.Tensor,q_tensor, p1, y1, p2, y2):
    residual_fp=residual_fp.to(EXT_type)
    assert residual_fp.is_cuda and residual_fp.is_contiguous()
    # n_elements = residual_fp.numel()
    # q_tensor = torch.empty((n_elements//8,1),dtype=torch.int32,device=residual_fp.device) 
    cuda_module.quant(residual_fp,q_tensor,p1,y1,p2-p1,y2,8-p2)
    return residual_fp
    #todo: triton implementation


@torch.no_grad()
def cuda_dequant_linear(residual_int): 
    #todo
    assert residual_int.is_contiguous()
    if not residual_int.is_cuda:
        return python_dequant_linear(residual_int) 
    n_elements = residual_int.numel()
    fp_tensor = torch.empty((n_elements,8),device=residual_int.device,dtype = EXT_type) 
    cuda_module.uniform_dequant(residual_int,fp_tensor)
    return fp_tensor

@torch.no_grad()
def cuda_dequant_3fold(residual_int, p1, y1, p2, y2) :
    assert residual_int.is_cuda and residual_int.is_contiguous()
    n_elements = residual_int.numel()
    fp_tensor = torch.empty((n_elements,8),device=residual_int.device,dtype = EXT_type) 
    cuda_module.dequant(residual_int,fp_tensor,p1,y1,p2-p1,y2,8-p2)
    return fp_tensor

@torch.no_grad()
def python_quant_linear(residual_fp): 
    delta_data = (residual_fp +0.5)*16
    noise = delta_data.new(delta_data.shape).uniform_(-0.5, 0.5)
    delta_data.add_(noise)
    delta_data.round_()
    ind = delta_data > 15.9 # ==16
    delta_data[ind] = 0.
    adj_overflow = torch.zeros_like(residual_fp)
    adj_overflow[ind] = 1.
    delta_data = delta_data.view(-1,8).to(torch.int32)
    delta_data[:,0] = (delta_data[:,0] << 28) + (delta_data[:,1] << 24) + (delta_data[:,2] << 20) + (delta_data[:,3] << 16) + \
                        (delta_data[:,4] << 12) + (delta_data[:,5] << 8) + (delta_data[:,6] << 4) + (delta_data[:,7])
    return delta_data[:,0].contiguous(), adj_overflow


@torch.no_grad()
def python_dequant_linear(residual_int):
    numel_h = residual_int.shape[0]
    residual_fp = torch.empty((numel_h,8),device=residual_int.device)
    for i in range(8):
        residual_fp[:,7-i] = (residual_int >> 4*i) % 16
    residual_fp = residual_fp/16 - 0.5
    return residual_fp

class ResidualTensor(object):
    def __init__(self, qshape,oshape):
        self.weight_scale = None
        self.qshape=qshape
        self.oshape=oshape
        self.lock_mu = self.mu = 0
        numel_h = prodx(oshape)
        assert numel_h % 8 == 0 #"residual shape must be multiple of 8"
        self.residual_int = torch.empty((numel_h//8,1),dtype=torch.int32,device='cuda')
        self.name = 'residual'
        self.init = True


    def set_value(self, residual_fp, weight_scale=None):
        p1, y1, p2, y2 = three_fold_curve_4bit(self.mu)
        adj = cuda_quant_3fold(residual_fp,self.residual_int, p1, y1, p2, y2)
        if weight_scale is not None:
            self.weight_scale = weight_scale

        self.lock_mu = self.mu
        self.init = False
        return adj

    def update_mu(self, optmizer_std):
        mu = (self.weight_scale/(optmizer_std+1e-8)).mean().item()
        if mu>75.8:
            self.mu = 3
        if mu>36.5:
            self.mu = 2
        elif mu > 16.2:
            self.mu = 1
        else:
            self.mu = 0
        if self.mu != self.lock_mu:
            print(f'{self.name} now mu level is {mu}({self.mu}), pre is {self.lock_mu}')

    def dequant(self):
        p1, y1, p2, y2 = three_fold_curve_4bit(self.lock_mu)
        residual_fp = cuda_dequant_3fold(self.residual_int, p1, y1, p2, y2)
        return residual_fp

    def add(self, tensor: torch.Tensor):
        Tdtype = tensor.dtype
        if not self.init:
            ret = (self.dequant().view(self.qshape) * self.weight_scale).to(dtype=Tdtype).view(self.oshape).add(tensor)
        else:
            ret = tensor
        return ret


    def __repr__(self):
        return "ResidualTensor !"
    



class GradientQTensor(object):
    def __init__(self,shape , bit=8):
        self.acc_ite = 0
        self.bit = bit
        self.qdata = 0
        self.scale = None
        self.qzero = None
        self.shape = shape
        self.fixscale = False
        self.acc_ite = 0
        #todo: random or determined quantizer
        # self.random = True

    def find_param(self,data):
        # self.vmin = data.amin(dim=1,keepdim=True)
        self.vmax = data.abs().amax(dim=1,keepdim=True)*4 ##gradient accumulation//2
        self.scale = torch.clamp((self.vmax ), min=1e-5) / (2 ** (self.bit-1) - 1)
        self.scale = self.scale.to(data.dtype) 
        # self.qzero = (-self.vmin / self.scale).round()


    def quant(self,fdata):
        scale = self.scale
        noise = fdata.new(fdata.shape).uniform_(-0.5, 0.5)
        qdata = torch.round(fdata / scale  + noise)
        self.qdata = torch.clamp(qdata, -2 ** (self.bit - 1), 2 ** (self.bit-1) - 1).to(torch.int8)

    # def __eq__(self, other):
    #     return self.qdata == other

    def dequant(self):
        scale = self.scale
        return (self.qdata.to(self.fdtype)) * (scale)

    def set_value(self, fdata):
        if self.bit > 8:
            self.qdata = fdata
            return
        if not (self.fixscale and self.scale is None):
            self.find_param(fdata)
        self.quant(fdata)
        self.fdtype = fdata.dtype

        self.acc_ite = 1

    def zero(self):
        self.acc_ite = 0
        self.qdata = None

    def add_(self, fdata):
        if self.bit > 8:
            self.qdata += fdata
        else:
            odata = self.dequant()
            self.acc_ite += 1
            self.quant(odata+fdata)
            del odata,fdata

    def get_value(self):
        if self.bit > 8:
            return self.qdata
        return self.dequant()

    def __repr__(self):
        return f"GradientQTensor bitwidth {self.bit} !"
    
    def to(self, device):
        self.qdata = self.qdata.to(device)
        if self.bit <= 8:
            self.scale = self.scale.to(device)
            self.qzero = self.qzero.to(device)
        return self

    @property
    def device(self):
        return self.qdata.device
    


import math
import warnings
class MyAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),obit=8,mu_updata_f =20,cpu=False):
        super().__init__(params, lr=lr, betas=betas)
        self.initialized = False
        self.obit = obit
        self.mu_updata_f = mu_updata_f
        self.statedevice = 'cpu' if cpu else 'cuda'
    def add_param_group(self, param_group):
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)


        for name, default in self.defaults.items():
            param_group.setdefault(name, default)

        params = param_group['params']

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()


        #if self.is_paged: self.page_mng.prefetch_all()
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    self.init_state(group, p, gindex, pindex)
                self.update_step(group, p, gindex, pindex)
                torch.cuda.synchronize()


        return loss

    def to_gpu(self):
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p in self.state:
                    values = self.state[p]
                    for k, v in values.items():
                        if isinstance(v, torch.Tensor):
                            is_paged = getattr(v, 'is_paged', False)
                            if not is_paged:
                                self.state[p][k] = v.to(p.device)

    def get_config(self, gindex, pindex, group):
        config = {}
        config["betas"] = group["betas"]
        config["eps"] = group["eps"]
        config["weight_decay"] = group["weight_decay"]
        config["lr"] = group["lr"]
        config["block_wise"] = True

        return config

    def get_state_buffer(self, pshape,pdevice, dtype=torch.float32):
        return torch.zeros(pshape, dtype=dtype, device=pdevice)

    @torch.no_grad()
    def init_state(self, group, p, gindex, pindex):
        config = self.get_config(gindex, pindex, group)

        if self.obit == 8:
            dtype = torch.uint8
        else:
            dtype = torch.bfloat16

        if p.numel() < 65536:
            #todo: adaptive dtype or from argument
            dtype = torch.bfloat16

        state = self.state[p]
        state["step"] = 0

        if dtype == torch.bfloat16:
            state["state1"] = self.get_state_buffer(p.shape,self.statedevice, dtype=dtype)
            state["state2"] = self.get_state_buffer(p.shape,self.statedevice, dtype=dtype)
        else:
            # assert p.numel() % 128==0 
            state["state1"] = self.get_state_buffer(p.shape,self.statedevice, dtype=torch.uint8)
            state["state2"] = self.get_state_buffer(p.shape,self.statedevice, dtype=torch.uint8)

            if config["block_wise"]:
                n = p.numel()
                if p.numel() % 2048==0:
                    blocks = n // 2048
                else:
                    assert p.numel() % 128==0
                    blocks = n // 128
                # blocks = min(blocks, 2048)
            else:
                blocks = 1
            state["absmax1"] = torch.zeros(
                (blocks,1), dtype=torch.float32, device=p.device
            )
            state["qmap1"] = torch.zeros(
                (blocks,1), dtype=torch.float32, device=p.device
            ) + 1e-10
            state["absmax2"] = torch.zeros(
                (blocks,1), dtype=torch.float32, device=p.device
            )
            state["qmap2"] = torch.zeros(
                (blocks,1), dtype=torch.float32, device=p.device
            ) + 1e-10

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        state = self.state[p]
        config = self.get_config(gindex, pindex, group)
        grad = p.grad       
        if isinstance(p,QuantTensor):
            if isinstance(grad, GradientQTensor):
                grad = grad.get_value()
            if isinstance(p.residual,ResidualTensor):
                if p.wbit == 8:
                    pdata = p.residual.add((p.data*p.qparam).to(grad.dtype))
                elif p.wbit == 4:
                    pdata = p.residual.add(DeQfunc_4bit_g(p.data,p.qparam,grad.dtype))
                else:
                    assert False
            else:
                pdata = p.residual.data.to(grad.dtype)
        else:
            pdata = p.data

        if state["state1"].dtype == torch.uint8:
            # s1,s2 = torch.zeros_like(grad), torch.zeros_like(grad)
            s1 = DeQfunc_8bit_opt(state["state1"].to(grad.device),[state["qmap1"],state["absmax1"]])
            exp_avg = s1.to(grad.dtype)
        else:
            exp_avg = state["state1"].to(grad.device)
        if state["state2"].dtype == torch.uint8:
            s2 = DeQfunc_8bit_opt(state["state2"].to(grad.device),[state["qmap2"],state["absmax2"]])
            exp_avg_sq = (s2*s2).to(grad.dtype)
        else:
            exp_avg_sq = (state["state2"]*state["state2"]).to(grad.device)

        beta1, beta2 = config["betas"]

        state["step"] += 1

        if config["weight_decay"] != 0:
            grad = grad.add(pdata, alpha=config["weight_decay"])

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)

        exp_avg_sqrt = exp_avg_sq.sqrt_()

        denom = exp_avg_sqrt.add_(group["eps"])

        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]
        step_size = config["lr"] * math.sqrt(bias_correction2) / bias_correction1

        pdata.addcdiv_(exp_avg, denom, value = -step_size)

        temp_data = torch.empty_like(pdata,dtype=torch.float32)

        if state["step"] > 250 and state["step"] % self.mu_updata_f == (self.mu_updata_f-1) and hasattr(p,'residual') and isinstance(p.residual,ResidualTensor):
            optmizer_std = torch.mean((abs(exp_avg)/(denom)).view(p.residual.qshape),dim=1,keepdim=True)*step_size
            p.residual.update_mu(optmizer_std)

        if state["state1"].dtype == torch.uint8:
            temp_data.data = exp_avg.float()
            state1, qparam = Qfunc_8bit_opt(temp_data,blocksize=len(state["absmax1"]))
            state["state1"] = state1.to(state["state1"].device)
            state["qmap1"], state["absmax1"] = qparam
        else:
            state["state1"] = exp_avg

        if state["state2"].dtype == torch.uint8:
            temp_data.data = exp_avg_sqrt.float()
            state2, qparam = Qfunc_8bit_opt(temp_data,blocksize=len(state["absmax2"]))
            state["state2"] = state2.to(state["state2"].device)
            state["qmap2"], state["absmax2"] = qparam
        else:
            state["state2"] = exp_avg_sq

        del exp_avg,exp_avg_sq,exp_avg_sqrt,denom
        torch.cuda.empty_cache()

        if isinstance(p,QuantTensor):
            if p.wbit == 8:
                qweight = Qfunc_8bit_res(pdata, p.residual, qscales = p.qparam)
            elif p.wbit == 4:
                qweight = Qfunc_4bit_g_res(pdata, p.residual, qparam = p.qparam)
            p.data = qweight
        if isinstance(grad, GradientQTensor):
            p.grad.zero()
        else:
            p.grad = None

    
