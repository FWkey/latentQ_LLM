import math
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import triton
    import triton.language as tl
    from . import custom_autotune

    # code based https://github.com/fpgaminer/GPTQ-triton
    @custom_autotune.autotune(
        configs=[
            triton.Config({
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            }, num_stages=4, num_warps=4),
            triton.Config({
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            }, num_stages=4, num_warps=4),
            triton.Config({
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            }, num_stages=4, num_warps=4),
            triton.Config({
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            }, num_stages=4, num_warps=4),
            triton.Config({
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            }, num_stages=4, num_warps=4),
            triton.Config({
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            }, num_stages=2, num_warps=8),
            triton.Config({
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 64,
                'GROUP_SIZE_M': 8
            }, num_stages=3, num_warps=8),
            triton.Config({
                'BLOCK_SIZE_M': 32,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 128,
                'GROUP_SIZE_M': 8
            }, num_stages=2, num_warps=4),
        ],
        key=['M', 'N', 'K'],
        nearest_power_of_two=True,
        prune_configs_by={
            'early_config_prune': custom_autotune.matmul248_kernel_config_pruner,
            'perf_model': None,
            'top_k': None,
        },
    )
    @triton.jit
    def matmul_248_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr, g_ptr, M, N, K, bits, maxq, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_scales, stride_zeros,
                          BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
        """
        Compute the matrix multiplication C = A x B.
        A is of shape (M, K) float16
        B is of shape (K//8, N) int32
        C is of shape (M, N) float16
        scales is of shape (G, N) float16
        zeros is of shape (G, N) float16
        g_ptr is of shape (K) int32 
        """
        infearure_per_bits = 32 // bits

        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_mask = (offs_am[:, None] < M)
        # b_ptrs is set up such that it repeats elements along the K axis 8 times
        b_ptrs = b_ptr + ((offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        g_ptrs = g_ptr + offs_k
        # shifter is used to extract the N bits of each element in the 32-bit word from B
        scales_ptrs = scales_ptr + offs_bn[None, :]
        zeros_ptrs = zeros_ptr + (offs_bn[None, :] // infearure_per_bits)

        shifter = (offs_k % infearure_per_bits) * bits
        zeros_shifter = (offs_bn % infearure_per_bits) * bits
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, num_pid_k):
            g_idx = tl.load(g_ptrs)

            # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
            scales = tl.load(scales_ptrs + g_idx[:, None] * stride_scales)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros = tl.load(zeros_ptrs + g_idx[:, None] * stride_zeros)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

            zeros = (zeros >> zeros_shifter[None, :]) & maxq
            zeros = (zeros + 1) & maxq

            a = tl.load(a_ptrs, mask=a_mask, other=0.)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
            b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

            # Now we need to unpack b (which is N-bit values) into 32-bit values
            b = (b >> shifter[:, None]) & maxq  # Extract the N-bit values
            b = (b - zeros) * scales  # Scale and shift

            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
            g_ptrs += BLOCK_SIZE_K

        c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
        c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    @custom_autotune.autotune(configs=[
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 256,
            'GROUP_SIZE_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 128,
            'GROUP_SIZE_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 128,
            'GROUP_SIZE_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 128,
            'GROUP_SIZE_M': 8
        }, num_stages=2, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 8
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 8
        }, num_stages=2, num_warps=4),
    ],
                              key=['M', 'N', 'K'],
                              nearest_power_of_two=True)
    @triton.jit
    def transpose_matmul_248_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr, g_ptr, M, N, K, bits, maxq, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_scales,
                                    stride_zeros, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
        """
        Compute the matrix multiplication C = A x B.
        A is of shape (M, N) float16
        B is of shape (K//8, N) int32
        C is of shape (M, K) float16
        scales is of shape (G, N) float16
        zeros is of shape (G, N) float16
        g_ptr is of shape (K) int32 
        """
        infearure_per_bits = 32 // bits

        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_k
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_k = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        offs_n = tl.arange(0, BLOCK_SIZE_N)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_n[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
        a_mask = (offs_am[:, None] < M)
        # b_ptrs is set up such that it repeats elements along the K axis 8 times
        b_ptrs = b_ptr + ((offs_bk[:, None] // infearure_per_bits) * stride_bk + offs_n[None, :] * stride_bn)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        g_ptrs = g_ptr + offs_bk
        g_idx = tl.load(g_ptrs)

        # shifter is used to extract the N bits of each element in the 32-bit word from B
        scales_ptrs = scales_ptr + offs_n[None, :] + g_idx[:, None] * stride_scales
        zeros_ptrs = zeros_ptr + (offs_n[None, :] // infearure_per_bits) + g_idx[:, None] * stride_zeros

        shifter = (offs_bk % infearure_per_bits) * bits
        zeros_shifter = (offs_n % infearure_per_bits) * bits
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

        for n in range(0, num_pid_n):
            # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
            scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

            zeros = (zeros >> zeros_shifter[None, :]) & maxq
            zeros = (zeros + 1) & maxq

            a = tl.load(a_ptrs, mask=a_mask, other=0.)  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
            b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

            # Now we need to unpack b (which is N-bit values) into 32-bit values
            b = (b >> shifter[:, None]) & maxq  # Extract the N-bit values
            b = (b - zeros) * scales  # Scale and shift
            b = tl.trans(b)

            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_N
            b_ptrs += BLOCK_SIZE_N
            scales_ptrs += BLOCK_SIZE_N
            zeros_ptrs += (BLOCK_SIZE_N // infearure_per_bits)

        c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bk[None, :]
        c_mask = (offs_am[:, None] < M) & (offs_bk[None, :] < K)
        tl.store(c_ptrs, accumulator, mask=c_mask)
except:
    print('triton not installed.')


def matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    with torch.cuda.device(input.device):
        output = torch.empty((input.shape[0], qweight.shape[1]), device=input.device, dtype=torch.float16)
        grid = lambda META: (triton.cdiv(input.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(qweight.shape[1], META['BLOCK_SIZE_N']), )
        matmul_248_kernel[grid](input, qweight, output, scales, qzeros, g_idx, input.shape[0], qweight.shape[1], input.shape[1], bits, maxq, input.stride(0), input.stride(1), qweight.stride(0),
                                qweight.stride(1), output.stride(0), output.stride(1), scales.stride(0), qzeros.stride(0))
        return output


def transpose_matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    with torch.cuda.device(input.device):
        output_dim = (qweight.shape[0] * 32) // bits
        output = torch.empty((input.shape[0], output_dim), device=input.device, dtype=torch.float16)
        grid = lambda META: (triton.cdiv(input.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(output_dim, META['BLOCK_SIZE_K']), )
        transpose_matmul_248_kernel[grid](input, qweight, output, scales, qzeros, g_idx, input.shape[0], qweight.shape[1], output_dim, bits, maxq, input.stride(0), input.stride(1), qweight.stride(0),
                                          qweight.stride(1), output.stride(0), output.stride(1), scales.stride(0), qzeros.stride(0))
        return output


class QuantLinearFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, maxq):
        output = matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq = bits, maxq
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq = ctx.bits, ctx.maxq
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = transpose_matmul248(grad_output, qweight, scales, qzeros, g_idx, bits, maxq)
        return grad_input, None, None, None, None, None, None


class QuantLinearGPTQ(nn.Module):

    def __init__(self, bits, groupsize, infeatures, outfeatures, bias):
        super().__init__()
        if bits not in [2, 4, 8]:
            raise NotImplementedError("Only 2,4,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.maxq = 2**self.bits - 1
        self.groupsize = groupsize if groupsize != -1 else infeatures

        self.register_buffer('qweight', torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32))
        self.register_buffer('qzeros', torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures // 32 * self.bits), dtype=torch.int32))
        self.register_buffer('scales', torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures), dtype=torch.float16))
        self.register_buffer('g_idx', torch.tensor([i // self.groupsize for i in range(infeatures)], dtype=torch.int32))
        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

    def unpack(self, tensor, scales, zeros, g_idx=None):
        k = 32 // self.bits
        g_n = zeros.shape[0]
        fweight = torch.zeros(self.outfeatures, self.infeatures,device=tensor.device)
        intweight = torch.zeros(self.infeatures, self.outfeatures,dtype=torch.int8,device=tensor.device)
        zeros_all = torch.zeros((zeros.shape[0],zeros.shape[1],k),dtype=torch.int8,device=tensor.device)
        for i in range(k):
            zeros_all[:,:,i] = (zeros >> (i*self.bits)) & self.maxq
        zeros_all = zeros_all.view(zeros.shape[0],-1)[:,:self.outfeatures]
        for n_cin in range(self.infeatures):
            ki = n_cin //k
            kj = n_cin%k
            intweight[n_cin] = (tensor[ki] >> (kj*self.bits) & self.maxq)
        for n_cin in range(self.infeatures):
            fweight[:,n_cin] = ((intweight[n_cin] - zeros_all[self.g_idx[n_cin]] - 1)) * scales[self.g_idx[n_cin]]
        # self.fweight = fweight
        return fweight

    def pack(self, linear, scales, zeros, g_idx=None):
        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(torch.round((linear.weight.data[:, idx] + scale_zeros[self.g_idx[idx]]) / self.scales[self.g_idx[idx]]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros((intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
        i = 0
        row = 0
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros -= 1
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")

        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)
        # fweight = self.unpack(self.qweight,self.scales,self.qzeros)

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures, )
        out = QuantLinearFunction.apply(x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, self.g_idx, self.bits, self.maxq)
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    
QuantLinear0 = QuantLinearGPTQ

@torch.no_grad()
def python_compress(fdata,bit=4):
    assert bit==4
    int_data = fdata.view(-1,8//bit).to(torch.int8)
    # delta_data[:,0] = delta_data[:,0] << 28 + delta_data[:,1] << 24 + delta_data[:,2] << 20 + delta_data[:,3] << 16 + \
    #                     delta_data[:,4] << 12 + delta_data[:,5] << 8 + delta_data[:,6] << 4 + delta_data[:,7]
    int_data[:,0] =  (int_data[:,0] << 4) + int_data[:,1]
    return int_data[:,0].contiguous()


@torch.no_grad()
def python_decompress(int_data,bit=4):
    assert bit==4
    numel_h = int_data.shape[0]
    fdata = torch.empty((numel_h,2),device=int_data.device)
    # for i in range(8):
    #     residual_fp[:,7-i] = (residual_int >> 4*i) % 16
    fdata[:,0] = (int_data >> 4) % 16
    fdata[:,1] = (int_data) % 16
    return fdata

class QuantLinearA16W4(nn.Module):

    def __init__(self, bits, groupsize, infeatures, outfeatures, bias):
        super().__init__()
        if bits not in [2, 4, 8]:
            raise NotImplementedError("Only 2,4,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.maxq = 2**self.bits - 1
        self.fweight = None
        self.groupsize = groupsize if groupsize != -1 else infeatures
        assert infeatures % 32 == 0

        self.register_buffer('qweight', torch.zeros((infeatures // 8 * self.bits*outfeatures), dtype=torch.int32))
        self.register_buffer('zeros', torch.zeros((math.ceil(infeatures / self.groupsize)*outfeatures,1), dtype=torch.float16))
        self.register_buffer('scales', torch.zeros((math.ceil(infeatures / self.groupsize)*outfeatures,1), dtype=torch.float16))
        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None



    def unpack(self, tensor, g_idx=None):
        fintweight = python_decompress(tensor).view(-1, self.infeatures)
        if g_idx is None:
            fintweight = fintweight.view(-1, self.groupsize)
            fweight = (fintweight - self.zeros.to(tensor.device))*self.scales.to(tensor.device)
        else:
            raise NotImplementedError
        return fweight.view(self.outfeatures,self.infeatures).half()


    def pack(self, linear, scales, zeros, g_idx=None):
        # if g_idx is not None:
        #     scales = scales[g_idx]
        #     zeros = zeros[g_idx]
        scales = scales.contiguous().half().reshape(-1, 1)
        self.scales = scales
        zeros = zeros.contiguous().half().reshape(-1, 1)
        self.zeros = zeros
        self.fweight = linear.weight.data.cpu().clone()
        # scale_zeros = zeros * scales
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()
        linear.weight.data = linear.weight.data.contiguous()
        # org_shape = linear.weight.data.shape
        intweight = torch.round((linear.weight.data.view(-1, self.groupsize).float() ) / self.scales+self.zeros).to(torch.int)
        self.qweight = python_compress(intweight)
        # ffdata = self.unpack(self.qweight)

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures, )
        out = x.reshape(-1, x.shape[-1])@ self.unpack(self.qweight).t()
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)

QuantLinear2 = QuantLinearA16W4




# Inherit from Function
class LinearFunction(torch.autograd.Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, weight, bias):
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        ctx.save_for_backward(input, weight, bias)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias



@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t = t.squeeze().view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().clamp_(min=-q_max-1,max=q_max).mul_(scales)
    return t.view(t_shape)


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t = t.squeeze().view(-1, t_shape[-1])
    if t.shape[1] > 10:
        maxs = t.abs().max(dim=0)[0]
        maxs = maxs.sort()[0]
        scales = min(maxs[-10]*2, maxs[-1])
    else:
        scales = t.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().clamp_(min=-q_max-1,max=q_max).mul_(scales)
    return t.view(t_shape)

@torch.no_grad()
def quantize_activation_per_tensor_asym(t, n_bits=8):
    t_shape = t.shape
    t = t.squeeze().view(-1, t_shape[-1])
    if len(t) > 10:
        minv = t[3:].min()
        maxv = t[3:].max()
    else:
        minv = t.min()
        maxv = t.max()
    q_max = 2**(n_bits)-1
    scales = (maxv-minv).clamp_(min=1e-5).div_(q_max)
    t -= minv
    t.div_(scales).round_().clamp_(min=0,max=q_max).mul_(scales).add_(minv)
    return t.view(t_shape)

@torch.no_grad()
def quantize_activation_static(t, absmax, n_bits=8):
    # t_shape = t.shape
    # t.view(-1, t_shape[-1])
    # scales = t.abs().max()
    q_max = 2**(n_bits-1)-1
    scale = absmax/q_max
    t.div_(scale).round_().clamp_(-q_max,q_max).mul_(scale)
    return t

@torch.no_grad()
def ident(t):
    return t

from functools import partial
from torch_int._CUDA import linear_a8_w8_bfp16_ofp16,linear_a8_w8_bfp32_ofp32,linear_a8_w8_bbf16_obf16
class QuantLinearA8W8(nn.Module):

    def __init__(self, bits, groupsize, infeatures, outfeatures, bias):
        super().__init__()
        if bits not in [2, 4, 8]:
            raise NotImplementedError("Only 2,4,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.maxq = 2**self.bits - 1
        self.fweight = None
        assert groupsize == -1
        groupsize = infeatures
        # assert infeatures % 32 == 0
        self.name = ''
        # self.inp_absmax = inp_absmax

        self.act_quant = lambda x:x
        # self.act_quant = quantize_activation_per_tensor_absmax
        # self.act_quant = quantize_activation_per_token_absmax
        self.register_buffer('inp_absmax', torch.zeros((self.infeatures), dtype=torch.float16))
        self.register_buffer('out_scale', torch.zeros((outfeatures), dtype=torch.float16))
        self.input_bias = None
        self.register_buffer('qweight', torch.zeros((outfeatures, infeatures // 8 * self.bits), dtype=torch.uint8))
        self.register_buffer('qzeros', torch.zeros((outfeatures,1), dtype=torch.int16))
        self.register_buffer('scales', torch.zeros((outfeatures,1), dtype=torch.float16))
        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None
        self.clampv = None

    def set_clampv(self,up_scale=1.0):
        self.inp_absmax *= up_scale
        self.clampv = self.inp_absmax.max().item()
        self.act_quant = partial(quantize_activation_static,absmax=self.clampv)

    def modify_clampv(self,up_scale=1.0):
        self.clampv = self.inp_absmax.max().item() * up_scale
        self.act_quant = partial(quantize_activation_static,absmax=self.clampv)

    def shutdown_actquant(self):
        self.act_quant = lambda x:x
        self.clampv = None

    def turnon_static_actquant(self,up_scale=1.0):
        self.clampv = self.inp_absmax.max().item() * up_scale
        self.act_quant = partial(quantize_activation_static,absmax=self.clampv)

    def config_act_func(self,config):
        self.outfeatures = len(self.scales)
        self.infeatures = len(self.inp_absmax)
        if callable(config):
            self.act_quant = config
        elif 'static' in config:
            # self.register_buffer('act_scale', torch.tensor(6., dtype=torch.float16))
            clamp = self.inp_absmax.max()
            self.clampv = clamp.item()
            self.act_quant = partial(quantize_activation_static,absmax=self.clampv)

            # if 'dense_4h_to_h' in self.name:
            #     # layeri = int(self.name.split('.')[2])
            #     self.clampv = clamp.item()
            #     self.act_quant = partial(quantize_activation_static,absmax=self.clampv/2)
            #     # self.act_quant = lambda x:x

        elif 'per_tensor' in config:
            self.act_quant = quantize_activation_per_tensor_absmax

        elif 'per_token' in config:
            self.act_quant = quantize_activation_per_token_absmax
        else:
            self.act_quant = lambda x:x

    def unpack(self, tensor):
        k = 32 // self.bits
        # fweight = torch.zeros(self.outfeatures, self.infeatures, dtype=torch.float16)
        if self.bits == 8:
            fweight = (tensor - self.qzeros)*self.scales
        elif self.bits == 4:
            intweight = torch.zeros(self.outfeatures, self.infeatures//2,2, dtype=torch.int8,device = tensor.device)
            intweight[:,:,1] = (tensor&15)
            intweight[:,:,0] = ((tensor >> 4)&15)
            intweight = intweight.view(tensor.shape[0],-1)
            fweight = (intweight - self.qzeros)*self.scales
        elif self.bits == 2:
            assert 0
        return fweight.to(self.scales.dtype)


    def pack(self, linear, scales, zeros, g_idx=None):

        scales = scales.reshape(-1,1)
        # self.linear0 = linear.cpu()
        # self.fweight = linear.weight.data.cpu().clone()
        zeros = zeros.reshape(-1,1)
        scale_zeros = zeros * scales
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        intweight = torch.round((linear.weight.data + scale_zeros)/ self.scales).to(torch.uint8)
        if self.bits == 8:
            self.qweight = intweight
        elif self.bits == 4:
            intweight = intweight.view(intweight.shape[0],-1,2)
            self.qweight = (intweight[:,:,0] << 4) + intweight[:,:,1]
        elif self.bits == 2:
            assert False


        self.qzeros.data = zeros.to(dtype=torch.int16)
        if True:
            ffdata = self.unpack(self.qweight)

    def forward(self, x):
        # clampv = self.inp_absmax.max().item()
        if self.input_bias is not None:
            x -= self.input_bias
        clampv = self.clampv
        if clampv is not None:
            x.clamp_(min=-clampv,max=clampv)
        out_shape = x.shape[:-1] + (self.outfeatures, )
        #q_x = self.act_quant(x).to(self.scales.dtype)
        #out = LinearFunction.apply(q_x.reshape(-1, x.shape[-1]), self.unpack(self.qweight), self.bias)
        Am = (x*127./clampv).round().clamp(-127,127).view(-1, x.shape[-1]).to(torch.int8)
        Bm = (self.qweight - self.qzeros).to(torch.int8)
        #out = linear_a8_w8_bfp16_ofp16( Am, Bm, self.bias, self.scales*clampv/127.)
        try:
             if self.bias:
                 out = linear_a8_w8_bbf16_obf16( Am, Bm, self.bias/self.scales.view(1,-1), clampv/127.)*self.scales.view(1,-1)
             else:
                 out = linear_a8_w8_bbf16_obf16( Am, Bm, 0*self.scales.view(1,-1), clampv/127.)*self.scales.view(1,-1)
        except:
             q_x = self.act_quant(x).to(self.scales.dtype)
             out = LinearFunction.apply(q_x.reshape(-1, x.shape[-1]), self.unpack(self.qweight), self.bias)
        #import pdb
        #pdb.set_trace()
        if self.out_scale.max() > 0:
            out = quantize_activation_static(out , self.out_scale, n_bits=8)
        return out.reshape(out_shape).to(x.dtype)

        
QuantLinear1 = QuantLinearA8W8



config = {'method':'gptq'}

import torch.nn.functional as F
def make_quant_linear(module, quantizers, bits, groupsize, name=''):
    if isinstance(module, QuantLinearA8W8) or isinstance(module, QuantLinearGPTQ) or isinstance(module, QuantLinearA16W4):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in quantizers:
            delattr(module, attr)
            if groupsize == -1:
                newlayer = QuantLinearA8W8(bits, groupsize, tmp.in_features, tmp.out_features, tmp.bias is not None)
                newlayer.name = name1
                if hasattr(tmp,'qk_absmax'):
                    newlayer.out_scale.data = tmp.qk_absmax.max()
                if hasattr(tmp,'v_absmax'):
                    newlayer.out_scale.data = tmp.v_absmax
                if hasattr(tmp,'inp_absmax'):
                    newlayer.inp_absmax.data = tmp.inp_absmax
                if hasattr(tmp,'inp_bias'):
                    delattr(newlayer,'input_bias')
                    newlayer.register_buffer('input_bias',tmp.inp_bias)
                    tmp_bias = F.linear(tmp.inp_bias.float(), tmp.weight.float()).to(torch.float16)
                    newlayer.tmp_bias = tmp_bias

                setattr(module, attr, newlayer)

            else:
                if config['method'] == 'gptq':
                    newlayer =QuantLinearGPTQ(bits, groupsize, tmp.in_features, tmp.out_features, tmp.bias is not None)
                else:
                    newlayer =QuantLinearA16W4(bits, groupsize, tmp.in_features, tmp.out_features, tmp.bias is not None)
                newlayer.name = name1
                if hasattr(tmp,'qk_absmax'):
                    newlayer.out_scale.data = tmp.qk_absmax.max()
                if hasattr(tmp,'v_absmax'):
                    newlayer.out_scale.data = tmp.v_absmax
                setattr(module, attr, newlayer)
    for name1, child in module.named_children():
        make_quant_linear(child, quantizers, bits, groupsize, name + '.' + name1 if name != '' else name1)





def autotune_warmup_linear(model, transpose=False):
    """
    Pre-tunes the quantized kernel
    """
    from tqdm import tqdm

    kn_values = {}

    for _, m in model.named_modules():
        if not isinstance(m, QuantLinearGPTQ):
            continue

        k = m.infeatures
        n = m.outfeatures

        if (k, n) not in kn_values:
            kn_values[(k, n)] = (m.qweight.cuda(), m.scales.cuda(), m.qzeros.cuda(), m.g_idx.cuda(), m.bits, m.maxq)

    print(f'Found {len(kn_values)} unique KN Linear values.')

    print('Warming up autotune cache ...')
    with torch.no_grad():
        for m in tqdm(range(0, 12)):
            m = 2**m  # [1, 2048]
            for (k, n), (qweight, scales, qzeros, g_idx, bits, maxq) in kn_values.items():
                a = torch.randn(m, k, dtype=torch.float16, device='cuda')
                matmul248(a, qweight, scales, qzeros, g_idx, bits, maxq)
                if transpose:
                    a = torch.randn(m, n, dtype=torch.float16, device='cuda')
                    transpose_matmul248(a, qweight, scales, qzeros, g_idx, bits, maxq)
    del kn_values
