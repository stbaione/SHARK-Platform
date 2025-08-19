# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Signatures for dynamic dispatch of ops covering our fundamental tensor types."""

from typing import Optional, Sequence, Union, List, Tuple
from numbers import Number, Integral
import math

import torch
from torch import Tensor, dtype

from sharktank.types import (
    AnyTensor,
    BlockScaledPackedLayout,
    QuantizedLayout,
    QuantizerTensor,
    Slice,
    ShardedTensor,
    SplitPrimitiveTensor,
    Theta,
    sharding,
    InferenceTensor,
    PrimitiveTensor,
    UnnamedTensorName,
)


from ._registry import *

__all__ = [
    "all_gather",
    "all_reduce",
    "argmax",
    "barrier_on_logical_device",
    "cat",
    "conv2d",
    "conv3d",
    "conv1d",
    "dequantize",
    "einsum_2args",
    "elementwise",
    "embedding_lookup",
    "equal",
    "expand",
    "extract_slice",
    "flatten",
    "gather",
    "gelu_sigmoid_approximation",
    "gelu_tanh_approximation",
    "gemm",
    "group_norm_affine",
    "layer_norm",
    "index_copy_",
    "index_put_",
    "index_select",
    "interpolate",
    "linear",
    "masked_fill",
    "matmul",
    "mean",
    "module_register_buffer",
    "pad",
    "permute",
    "quantize",
    "rms_norm",
    "reduce_scatter",
    "repeat",
    "replicate",
    "reshape",
    "reshard",
    "reshard_split",
    "reshard_like",
    "scaled_dot_product_attention",
    "scatter_",
    "scatter_add",
    "sharded_cat",
    "sharded_sum",
    "sharded_gather",
    "shards",
    "sigmoid",
    "softmax",
    "split",
    "squeeze",
    "sum",
    "swiglu",
    "to",
    "topk",
    "trace_tensor",
    "transfer_to_logical_device",
    "transpose",
    "unflatten",
    "unpack",
    "unpack_qs",
    "unshard",
    "unsqueeze",
    "view",
    "view_as_complex",
    "view_as_real",
    "zeros_like",
]

IntOrSequenceInt = Union[int, Sequence[int]]


@overridable(is_trivially_replicable=False)
def all_gather(maybe_sharded: AnyTensor, *, dim: int | None = None) -> AnyTensor:
    "Gather/concatenate on all devices along dimension `dim`."
    ...


@all_gather.trampoline
def _all_gather_trampoline(
    d: SignatureDispatcher, maybe_sharded: AnyTensor, *, dim: int | None = None
):
    tensors = (maybe_sharded,)
    for override in d.find_overrides(tensors):
        result = override(maybe_sharded, dim=dim)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable(is_trivially_replicable=False)
def all_reduce(tensor: AnyTensor) -> AnyTensor:
    "Reduce on all devices."
    ...


@all_reduce.trampoline
def _all_reduce_trampoline(d: SignatureDispatcher, tensor: AnyTensor):
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def argmax(
    tensor: AnyTensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
    chunk_size: Optional[int] = None,
) -> AnyTensor:
    "Take argmax of the tensor"
    ...


@argmax.trampoline
def _argmax_trampoline(
    d: SignatureDispatcher,
    tensor: AnyTensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
    chunk_size: Optional[int] = None,
):
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor, dim, keepdim=keepdim, chunk_size=chunk_size)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def cat(tensors: Tuple[AnyTensor, ...] | List[AnyTensor], dim: int = 0) -> AnyTensor:
    ...


@cat.trampoline
def _cat_trampoline(
    d: SignatureDispatcher, tensors: Tuple[Tensor, ...] | List[Tensor], dim: int = 0
):
    for override in d.find_overrides(tensors):
        result = override(tensors, dim)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def conv2d(
    input: AnyTensor,
    weight: AnyTensor,
    bias: Optional[AnyTensor] = None,
    *,
    stride: IntOrSequenceInt = 1,
    padding: IntOrSequenceInt = 0,
    dilation: IntOrSequenceInt = 1,
    groups: IntOrSequenceInt = 1,
    accum_dtype: Optional[torch.dtype] = None,
):
    """Equivalent to torch.nn.functional.conv2d with enhancements:

    * Primitive weight/bias tensors will be promoted to the input dtype.
    """
    raise NotImplementedError


@conv2d.trampoline
def _conv2d_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    weight: AnyTensor,
    bias: Optional[AnyTensor] = None,
    *,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    accum_dtype: Optional[torch.dtype] = None,
):
    tensors = [input, weight]
    if bias is not None:
        tensors.append(bias)
    for override in d.find_overrides(tensors):
        result = override(
            input,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            accum_dtype=accum_dtype,
        )
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def conv3d(
    input: AnyTensor,
    weight: AnyTensor,
    bias: Optional[AnyTensor] = None,
    *,
    stride: IntOrSequenceInt = 1,
    padding: IntOrSequenceInt = 0,
    dilation: IntOrSequenceInt = 1,
    groups: IntOrSequenceInt = 1,
    accum_dtype: Optional[torch.dtype] = None,
):
    """Equivalent to torch.nn.functional.conv3d with enhancements:

    * Primitive weight/bias tensors will be promoted to the input dtype.
    """
    raise NotImplementedError


@conv3d.trampoline
def _conv3d_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    weight: AnyTensor,
    bias: Optional[AnyTensor] = None,
    *,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    accum_dtype: Optional[torch.dtype] = None,
):
    tensors = [input, weight]
    if bias is not None:
        tensors.append(bias)
    for override in d.find_overrides(tensors):
        result = override(
            input,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            accum_dtype=accum_dtype,
        )
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def conv1d(
    input: AnyTensor,
    weight: AnyTensor,
    bias: Optional[AnyTensor] = None,
    *,
    stride: IntOrSequenceInt = 1,
    padding: IntOrSequenceInt = 0,
    dilation: IntOrSequenceInt = 1,
    groups: IntOrSequenceInt = 1,
    accum_dtype: Optional[torch.dtype] = None,
):
    """Equivalent to torch.nn.functional.conv1d with enhancements:

    * Primitive weight/bias tensors will be promoted to the input dtype.
    """
    raise NotImplementedError


@conv1d.trampoline
def _conv1d_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    weight: AnyTensor,
    bias: Optional[AnyTensor] = None,
    *,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    accum_dtype: Optional[torch.dtype] = None,
):
    tensors = [input, weight]
    if bias is not None:
        tensors.append(bias)
    for override in d.find_overrides(tensors):
        result = override(
            input,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            accum_dtype=accum_dtype,
        )
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def dequantize(
    input: AnyTensor | QuantizedLayout | dict[str, AnyTensor],
    /,
    *,
    quantizer: AnyTensor | None = None,
    dtype: torch.dtype | None = None,
) -> AnyTensor:
    """Dequantize a tensor. The input may be a quantized tensor, layout or a
    dictionary of planes.

    In some cases it is allowed for a plane to be missing if a quantizer is given.
    E.g. when we have a StaticScaledQuantizer the scale plane is not required."""
    ...


@dequantize.trampoline
def _dequantize_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    /,
    *,
    quantizer: AnyTensor | None = None,
    dtype: torch.dtype | None = None,
) -> AnyTensor:
    dispatch_args = (input, quantizer)
    for override in d.find_overrides(dispatch_args):
        result = override(input, quantizer=quantizer, dtype=dtype)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable
def einsum_2args(
    input0: AnyTensor,
    input1: AnyTensor,
    einsum_str: str,
    *,
    accum_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Executes a given Einstein summation notation string on the provided tensors.

    Equivalent to:
    ```
    y = torch.einsum(einsum_str, input0, input1)
    ```
    """
    raise NotImplementedError


@einsum_2args.trampoline
def _einsum_trampoline(
    d: SignatureDispatcher, input0: AnyTensor, input1: AnyTensor, einsum_str: str
):
    tensors = (input0, input1)
    for override in d.find_overrides(tensors):
        result = override(input0, input1, einsum_str)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def elementwise(operator, *args, **kwargs) -> AnyTensor:
    """Applies an elementwise operator against arguments."""
    raise NotImplementedError


@elementwise.trampoline
def _elementwise_trampoline(d: SignatureDispatcher, operator, *args, **kwargs):
    tensors = []
    for a in args:
        if isinstance(a, (Tensor, InferenceTensor)):
            tensors.append(a)
        else:
            break
    for override in d.find_overrides(tensors):
        result = override(operator, *args, **kwargs)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def embedding_lookup(
    input: AnyTensor, embedding_matrix: AnyTensor, dtype: Optional[dtype]
) -> AnyTensor:
    """Performs the equivalent of F.embedding(input, embedding_matrix).

    Note that the default algorithm will unquantize the embedding_matrix to
    do the lookup, which is inefficient. Specializations should decompose
    this as appropriate for quantized arithmetic.
    """
    raise NotImplementedError


@embedding_lookup.trampoline
def _embedding_lookup_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    embedding_matrix: AnyTensor,
    dtype: Optional[dtype],
):
    tensors = (input, embedding_matrix)
    for override in d.find_overrides(tensors):
        result = override(input, embedding_matrix, dtype)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def empty_like(
    tensor: AnyTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool = False,
    memory_format: torch.memory_format = torch.preserve_format,
) -> AnyTensor:
    """See torch.zeros_like"""
    ...


@empty_like.trampoline
def _empty_like_trampoline(
    d: SignatureDispatcher,
    tensor: AnyTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool = False,
    memory_format: torch.memory_format = torch.preserve_format,
) -> AnyTensor:
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(
            tensor,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
            memory_format=memory_format,
        )
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable(is_trivially_replicable=False)
def equal(a: AnyTensor, b: AnyTensor) -> bool:
    """Compares 2 tensors for equality, such that they elements and dtype are equal.

    Overrides are matched first against both tensor types and failing that,
    then on just the first.
    Therefore, each first-only argument override must internally decide whether
    it can handle an equality check with an arbitrary b tensor.
    """
    ...


@equal.trampoline
def _equal_trampoline(d: SignatureDispatcher, a: AnyTensor, b: AnyTensor):
    # Try first more specific matching the 2 operands.
    tensors = (
        a,
        b,
    )
    for override in d.find_overrides(tensors):
        result = override(a, b)
        if result is not NotImplemented:
            return override, result

    # Less specific. Try matching only the first operand.
    tensors = (a,)
    for override in d.find_overrides(tensors):
        result = override(a, b)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def expand(tensor: AnyTensor, shape: List[int]) -> AnyTensor:
    """See torch.Tensor.expand"""
    ...


@expand.trampoline
def _expand_trampoline(
    d: SignatureDispatcher, tensor: AnyTensor, shape: List[int]
) -> AnyTensor:
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor, shape)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def extract_slice(
    tensor: AnyTensor,
    key: Slice,
) -> torch.Tensor:
    """Indexes the tensor using the key.

    Equivalent to:
    ```
    out = tensor[key]
    ```
    """
    raise NotImplementedError


@extract_slice.trampoline
def _extract_slice_trampoline(d: SignatureDispatcher, tensor: AnyTensor, key: Slice):
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor, key)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def flatten(input: AnyTensor, start_dim: int = 0, end_dim: int = -1) -> AnyTensor:
    """See torch.flatten"""
    ...


@flatten.trampoline
def _flatten_trampoline(
    d: SignatureDispatcher, input: AnyTensor, start_dim: int = 0, end_dim: int = -1
) -> AnyTensor:
    dispatch_args = (input,)
    for override in d.find_overrides(dispatch_args):
        result = override(input, start_dim, end_dim)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable
def gather(input: AnyTensor, dim: int, index: AnyTensor) -> AnyTensor:
    """See torch.gather"""
    ...


@gather.trampoline
def _gather_trampoline(
    d: SignatureDispatcher, input: AnyTensor, dim: int, index: AnyTensor
) -> AnyTensor:
    dispatch_args = (
        input,
        index,
    )
    for override in d.find_overrides(dispatch_args):
        result = override(input, dim, index)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


def gelu_sigmoid_approximation(input: AnyTensor) -> AnyTensor:
    """Applies GELU approximation that is fast but somewhat inaccurate.
    See: https://github.com/hendrycks/GELUs
    """
    return input * elementwise(torch.sigmoid, 1.702 * input)


def gelu_tanh_approximation(input: AnyTensor) -> AnyTensor:
    """Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    Approximation with tanh"""
    return (
        0.5
        * input
        * (
            1.0
            + elementwise(
                torch.tanh,
                math.sqrt(2.0 / math.pi)
                * (input + 0.044715 * elementwise(torch.pow, input, 3.0)),
            )
        )
    )


@overridable
def gemm(
    a: AnyTensor,
    b: AnyTensor,
    c: Optional[AnyTensor] = None,
    alpha: Optional[Union[Number, AnyTensor]] = None,
    beta: Optional[Union[Number, AnyTensor]] = None,
    transa: bool = False,
    transb: bool = False,
):
    """GEMM as defined by BLAS.
    `alpha*a*b + beta*c`
    If `c` is None it is the zero-filed tensor.
    """
    raise NotImplementedError


@gemm.trampoline
def _gemm_trampoline(
    d: SignatureDispatcher,
    a: AnyTensor,
    b: AnyTensor,
    c: Optional[AnyTensor] = None,
    alpha: Optional[Union[Number, AnyTensor]] = None,
    beta: Optional[Union[Number, AnyTensor]] = None,
    transa: bool = False,
    transb: bool = False,
):
    tensors = (a, b, c)
    for override in d.find_overrides(tensors):
        result = override(
            a=a, b=b, c=c, alpha=alpha, beta=beta, transa=transa, transb=transb
        )
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def group_norm_affine(
    input: AnyTensor, weight: AnyTensor, bias: AnyTensor, *, num_groups: int, eps: float
):
    """Equivalent to torch.nn.functional.group_norm(affine=True)."""
    raise NotImplementedError


@group_norm_affine.trampoline
def _group_norm_affine_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    weight: AnyTensor,
    bias: AnyTensor,
    *,
    num_groups: int,
    eps: float,
):
    tensors = (input, weight, bias)
    for override in d.find_overrides(tensors):
        result = override(input, weight, bias, num_groups=num_groups, eps=eps)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def index_copy_(
    inout: AnyTensor, dim: int, index: AnyTensor, tensor: AnyTensor
) -> AnyTensor:
    """See torch.Tensor.index_copy_"""
    ...


@index_copy_.trampoline
def _index_copy__trampoline(
    d: SignatureDispatcher,
    inout: AnyTensor,
    dim: int,
    index: AnyTensor,
    tensor: AnyTensor,
) -> AnyTensor:
    tensors = (inout, index, tensor)
    for override in d.find_overrides(tensors):
        result = override(inout, dim, index, tensor)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def index_put_(
    inout: AnyTensor, indices: Tuple[AnyTensor], values: AnyTensor
) -> AnyTensor:
    """See torch.Tensor.index_put_"""
    ...


@index_put_.trampoline
def _index_put__trampoline(
    d: SignatureDispatcher,
    inout: AnyTensor,
    indices: Tuple[AnyTensor],
    values: AnyTensor,
) -> AnyTensor:
    # We change the order for the variadic indices to be last.
    tensors = (inout, values, *indices)
    for override in d.find_overrides(tensors):
        result = override(inout, indices, values)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def index_select(tensor: AnyTensor, dim: int, index: AnyTensor) -> AnyTensor:
    """See torch.Tensor.index_select"""
    ...


@index_select.trampoline
def _index_select_trampoline(
    d: SignatureDispatcher, tensor: AnyTensor, dim: int, index: AnyTensor
) -> AnyTensor:
    tensors = (tensor, index)
    for override in d.find_overrides(tensors):
        result = override(tensor, dim, index)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def interpolate(
    input: AnyTensor,
    size: Optional[int | List[int]] = None,
    scale_factor: Optional[float | List[float]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> AnyTensor:
    """Equivalent to torch.nn.functional.interpolate"""
    raise NotImplementedError


@interpolate.trampoline
def _interpolate_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    size: Optional[int | List[int]] = None,
    scale_factor: Optional[float | List[float]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> AnyTensor:
    tensors = [input]
    for override in d.find_overrides(tensors):
        result = override(
            input,
            size,
            scale_factor,
            mode,
            align_corners,
            recompute_scale_factor,
            antialias,
        )
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def layer_norm(
    input: AnyTensor,
    weight: Optional[AnyTensor],
    bias: Optional[AnyTensor],
    *,
    eps: float,
    normalized_shape: Optional[tuple[int]] = None,
):
    """Equivalent to torch.nn.functional.layer_norm(elementwise_affine=True)."""
    raise NotImplementedError


@layer_norm.trampoline
def _layer_norm_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    weight: Optional[AnyTensor],
    bias: Optional[AnyTensor],
    *,
    eps: float,
    normalized_shape: Optional[tuple[int]] = None,
):
    tensors = [input]
    if weight is not None:
        tensors.append(bias)
    if bias is not None:
        tensors.append(bias)
    for override in d.find_overrides(tensors):
        result = override(
            input, weight, bias, eps=eps, normalized_shape=normalized_shape
        )
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def linear(
    input: AnyTensor,
    weight: AnyTensor,
    bias: Optional[AnyTensor] = None,
    *,
    accum_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Applies a linear transformation to the incoming data.

    Equivalent to:
    ```
    y = torch.matmul(input, weight.mT) + bias
    ```

    This operator is defined to operate on a limited number of quantized types.
    In that situation, the result may be a QuantizedTensor. Callers should
    be prepared to handle this scenario.

    The optional accum_dtype argument is used as a hint to some implementations
    which may need help in selecting an appropriate high precision type for
    accumulation.
    """
    raise NotImplementedError


@linear.trampoline
def _linear_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    weight: AnyTensor,
    bias: Optional[AnyTensor] = None,
    *,
    accum_dtype: Optional[torch.dtype] = None,
):
    tensors = (input, weight) if bias is None else (input, weight, bias)
    for override in d.find_overrides(tensors):
        result = override(input, weight, bias, accum_dtype=accum_dtype)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def masked_fill(input: AnyTensor, mask: AnyTensor, value: Number) -> AnyTensor:
    """See torch.masked_fill"""
    ...


@masked_fill.trampoline
def _masked_fill_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    mask: AnyTensor,
    value: Number,
) -> AnyTensor:
    tensors = (input, mask)
    for override in d.find_overrides(tensors):
        result = override(input, mask, value)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def matmul(lhs: AnyTensor, rhs: AnyTensor, *, transpose_rhs: bool = False):
    """Performs a matmul where the RHS may be an InferenceTensor.

    Unlike torch.matmul, this variant is optimized for emission of a fused
    `matmul(lhs, rhs.mT)` when `transpose_rhs=True`. Most inference optimizers
    will store their weights in this way and assume fusions that operate on them.

    Args:
    lhs: Left hand side tensor. Can have dimensionality > 2 for batch.
    rhs: Right hand side tensor. Must be 2d or a scalar.
    transpose_rhs: Whether the right hand side should be transposed prior
        to matmul.
    """
    raise NotImplementedError


@matmul.trampoline
def _matmul_trampoline(
    d: SignatureDispatcher, lhs, rhs, *, transpose_rhs: bool = False
):
    tensors = (lhs, rhs)
    for override in d.find_overrides(tensors):
        result = override(lhs, rhs, transpose_rhs=transpose_rhs)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def pad(
    input: AnyTensor, _pad: Sequence[int], mode: str, value: Optional[float]
) -> AnyTensor:
    """See torch.nn.functional.pad"""
    ...


@pad.trampoline
def _pad_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    _pad: Sequence[int],
    mode: str = "constant",
    value: Optional[float] = None,
) -> AnyTensor:
    if value is None:
        value = 0
    tensors = (input,)
    for override in d.find_overrides(tensors):
        result = override(input, _pad, mode=mode, value=value)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def permute(tensor: AnyTensor, dims: List[int]) -> AnyTensor:
    """Permute the tensor dimensions according to the permutation `dims` in line
    notation.
    The semantics are the same as torch.permute."""
    ...


@permute.trampoline
def _permute_trampoline(d: SignatureDispatcher, tensor: AnyTensor, dims: List[int]):
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor, dims)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def mean(
    x: AnyTensor,
    dim: Union[int, List[int]],
    keepdim: bool = False,
    *,
    dtype: torch.dtype = None,
) -> AnyTensor:
    """See torch.mean"""
    raise NotImplementedError


@mean.trampoline
def _mean_trampoline(
    d: SignatureDispatcher,
    x: AnyTensor,
    dim: Union[int, List[int]],
    keepdim: bool = False,
    *,
    dtype: torch.dtype = None,
) -> AnyTensor:
    tensors = (x,)
    for override in d.find_overrides(tensors):
        result = override(x, dim, keepdim, dtype=dtype)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable(is_trivially_replicable=False)
def module_register_buffer(
    module: torch.nn.Module, name: str, tensor: AnyTensor
) -> None:
    """Register the tensor into the module. See torch.nn.Module.register_buffer."""
    ...


@module_register_buffer.trampoline
def _module_register_buffer_trampoline(
    d: SignatureDispatcher, module: torch.nn.Module, name: str, tensor: AnyTensor
) -> None:
    args = (module, tensor)
    for override in d.find_overrides(args):
        result = override(module, name, tensor)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(args)


@overridable
def quantize(
    tensor: AnyTensor, quantizer: AnyTensor, name: str = UnnamedTensorName
) -> AnyTensor:
    """Quantize a tensor using the provided quantizer."""
    ...


@quantize.trampoline
def _quantize_trampoline(
    d: SignatureDispatcher,
    tensor: AnyTensor,
    quantizer: AnyTensor,
    name: str = UnnamedTensorName,
) -> AnyTensor:
    tensors = (tensor, quantizer)
    for override in d.find_overrides(tensors):
        result = override(tensor, quantizer, name)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable(is_trivially_replicable=False)
def reduce_scatter(tensor: AnyTensor, scatter_dim: int) -> AnyTensor:
    """Reduces then splits/scatters across the devices."""
    ...


@reduce_scatter.trampoline
def _reduce_scatter_trampoline(
    d: SignatureDispatcher, tensor: AnyTensor, scatter_dim: int
):
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor, scatter_dim)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def rms_norm(
    x: AnyTensor, weight: AnyTensor, *, epsilon: float, orig_dtype: torch.dtype
) -> AnyTensor:
    """Computes the full, unbiased RMS normalization of an input."""
    raise NotImplementedError


@rms_norm.trampoline
def _rms_norm_trampoline(
    d: SignatureDispatcher,
    x: AnyTensor,
    weight: AnyTensor,
    *,
    epsilon: float,
    orig_dtype: torch.dtype,
):
    tensors = (x, weight)
    for override in d.find_overrides(tensors):
        result = override(x, weight, epsilon=epsilon, orig_dtype=orig_dtype)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def repeat(input: AnyTensor, *sizes: List[int]) -> AnyTensor:
    """See torch.Tensor.repeat"""
    ...


@repeat.trampoline
def _repeat_trampoline(
    d: SignatureDispatcher, input: AnyTensor, *sizes: List[int]
) -> AnyTensor:
    dispatch_args = (input,)
    for override in d.find_overrides(dispatch_args):
        result = override(input, *sizes)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable
def replicate(
    input: AnyTensor, count: int, devices: tuple[int, ...] | None
) -> ShardedTensor:
    """Replicate across devices.

    Possibly reshards if required."""
    ...


@replicate.trampoline
def _replicate_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    count: int,
    devices: tuple[int, ...] | None = None,
) -> ShardedTensor:
    tensors = (input,)
    if isinstance(input, ShardedTensor):
        assert devices is None
    else:
        devices = devices if devices is not None else tuple(range(count))

    for override in d.find_overrides(tensors):
        result = override(input, count=count, devices=devices)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def scaled_dot_product_attention(
    q: AnyTensor,
    k: AnyTensor,
    v: AnyTensor,
    a: Optional[AnyTensor],
    is_causal: bool = False,
    scale: Optional[float] = None,
    softcap: Optional[float] = None,
    impl: Optional[str] = None,
) -> AnyTensor:
    """Computes the scaled dot product attention using QKV."""
    raise NotImplementedError


@scaled_dot_product_attention.trampoline
def _scaled_dot_product_attention(
    d: SignatureDispatcher,
    q: AnyTensor,
    k: AnyTensor,
    v: AnyTensor,
    a: Optional[AnyTensor],
    is_causal: bool = False,
    scale: Optional[float] = None,
    softcap: Optional[float] = None,
    impl: Optional[str] = None,
):
    tensors = (q, k, v, a)
    for override in d.find_overrides(tensors):
        result = override(
            q, k, v, a, is_causal=is_causal, scale=scale, softcap=softcap, impl=impl
        )
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def reshape(input: AnyTensor, shape: List[int]) -> AnyTensor:
    """Returns a tensor with the same data and number of elements as input, but with
    the specified shape.
    See torch.reshape.
    """
    ...


@reshape.trampoline
def _reshape_trampoline(d: SignatureDispatcher, input, shape) -> AnyTensor:
    dispatch_args = (input,)
    for override in d.find_overrides(dispatch_args):
        result = override(input, shape)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable(is_trivially_replicable=False)
def reshard(
    input: AnyTensor | Theta,
    spec: (
        sharding.TensorSharding | sharding.ThetaLayerSharding | sharding.ThetaSharding
    ),
) -> AnyTensor | Theta:
    """Reshard to the given specification.
    If a Theta is given then the tensor nesting is preserved,
    but the tensors are sharded according to the spec.
    """
    ...


@reshard.trampoline
def _reshard_trampoline(d: SignatureDispatcher, input, spec) -> ShardedTensor:
    dispatch_args = (input, spec)
    for override in d.find_overrides(dispatch_args):
        result = override(input, spec)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable(is_trivially_replicable=False)
def reshard_split(
    input: AnyTensor, *, dim: int, count: int, devices: tuple[int, ...] | None
) -> ShardedTensor:
    """Split `input` along `dim`.
    This does not mean that a sharded tensor is further sharded.
    It is not composition of sharding operations.
    """
    ...


@reshard_split.trampoline
def _reshard_split_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    dim: int,
    count: int,
    devices: tuple[int, ...] | None = None,
) -> ShardedTensor:
    tensors = (input,)
    if isinstance(input, (torch.Tensor, PrimitiveTensor)):
        devices = devices if devices is not None else tuple(range(count))
    else:
        assert devices is None

    for override in d.find_overrides(tensors):
        result = override(input, dim=dim, count=count, devices=devices)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable(is_trivially_replicable=False)
def reshard_like(input: AnyTensor, like: AnyTensor) -> AnyTensor:
    """Shard `input` the same way as `like`.

    This may require expensive resharding."""
    ...


@reshard_like.trampoline
def _reshard_like_trampoline(
    d: SignatureDispatcher, input: AnyTensor, like: AnyTensor
) -> AnyTensor:
    tensors = (
        input,
        like,
    )
    for override in d.find_overrides(tensors):
        result = override(input, like)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def scatter_(
    inout: AnyTensor,
    dim: int,
    index: AnyTensor,
    src: AnyTensor | Number,
    *,
    reduce: str = None,
):
    """
    See torch.Tensor.scatter_
    NOTE: Does not modify the inout tensor in place for ShardedTensors, will return copy.
    """
    ...


@scatter_.trampoline
def _scatter__trampoline(
    d: SignatureDispatcher,
    inout: AnyTensor,
    dim: int,
    index: AnyTensor,
    src: AnyTensor | Number,
    *,
    reduce: str = None,
) -> AnyTensor:
    dispatch_args = (inout, index, src)
    for override in d.find_overrides(dispatch_args):
        result = override(inout, dim, index, src, reduce=reduce)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable
def scatter_add(
    input: AnyTensor, dim: int, index: AnyTensor, src: AnyTensor
) -> AnyTensor:
    """
    See torch.scatter_add
    """
    ...


@scatter_add.trampoline
def _scatter_add_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    dim: int,
    index: AnyTensor,
    src: AnyTensor,
) -> AnyTensor:
    tensors = (input, index, src)
    for override in d.find_overrides(tensors):
        result = override(input, dim, index, src)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable(is_trivially_replicable=False)
def sharded_cat(maybe_sharded: AnyTensor):
    """Concats all shards along the sharding dimension.

    Does nothing if not sharded.
    """
    raise NotImplementedError


@sharded_cat.trampoline
def _sharded_cat_trampoline(d: SignatureDispatcher, maybe_sharded: AnyTensor):
    tensors = (maybe_sharded,)
    for override in d.find_overrides(tensors):
        result = override(maybe_sharded)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable(is_trivially_replicable=False)
def sharded_gather(input: AnyTensor, root_rank: int) -> list[AnyTensor]:
    """Gather the input tensor from all devices to the given device ordinal."""
    ...


@sharded_gather.trampoline
def _sharded_gather_trampoline(
    d: SignatureDispatcher, input: AnyTensor, root_rank: int
) -> AnyTensor:
    dispatch_args = (input,)
    for override in d.find_overrides(dispatch_args):
        result = override(input, root_rank)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable(is_trivially_replicable=False)
def shards(input: ShardedTensor | QuantizedLayout) -> list[AnyTensor | QuantizedLayout]:
    """Return the shards of a sharded tensor."""
    ...


@shards.trampoline
def _shards_trampoline(
    d: SignatureDispatcher, input: AnyTensor | QuantizedLayout
) -> list[AnyTensor | QuantizedLayout]:
    dispatch_args = (input,)
    for override in d.find_overrides(dispatch_args):
        result = override(input)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable(is_trivially_replicable=False)
def sharded_sum(maybe_sharded: AnyTensor, root_rank: int = 0) -> AnyTensor:
    """Reduce across the shards into a single device.

    root_rank:
        Rank of receiving device within the tensor devices.
        If sharded, `maybe_sharded.devices[root_rank]` is the destination.
    """
    ...


@sharded_sum.trampoline
def _sharded_sum_trampoline(
    d: SignatureDispatcher, maybe_sharded: AnyTensor, root_rank: int = 0
):
    tensors = (maybe_sharded,)
    for override in d.find_overrides(tensors):
        result = override(maybe_sharded, root_rank)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def sigmoid(tensor: AnyTensor) -> AnyTensor:
    """See torch.sigmoid"""
    ...


@sigmoid.trampoline
def _sigmoid_trampoline(d: SignatureDispatcher, tensor: AnyTensor) -> AnyTensor:
    dispatch_args = (tensor,)
    for override in d.find_overrides(dispatch_args):
        result = override(tensor)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable
def softmax(
    tensor: AnyTensor, dim: Optional[int] = None, dtype: Optional[torch.dtype] = None
) -> AnyTensor:
    """See torch.nn.functional.softmax"""
    ...


@softmax.trampoline
def _softmax_trampoline(
    d: SignatureDispatcher,
    tensor: AnyTensor,
    dim: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> AnyTensor:
    dispatch_args = [tensor]
    for override in d.find_overrides(dispatch_args):
        result = override(tensor, dim=dim, dtype=dtype)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable
def split(
    tensor: AnyTensor, split_size_or_sections: int | list[int], dim: int = 0
) -> tuple[AnyTensor, ...]:
    """See torch.split"""
    ...


@split.trampoline
def _split_trampoline(
    d: SignatureDispatcher,
    tensor: AnyTensor,
    split_size_or_sections: int | list[int],
    dim: int,
) -> tuple[AnyTensor, ...]:
    dispatch_args = [tensor]
    for override in d.find_overrides(dispatch_args):
        result = override(tensor, split_size_or_sections, dim)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable
def swiglu(
    tensor: AnyTensor, *, alpha: float = 1.702, limit: float | None = None
) -> AnyTensor:
    raise NotImplementedError


@swiglu.trampoline
def _swiglu_trampoline(
    d: SignatureDispatcher,
    tensor: AnyTensor,
    *,
    alpha: float = 1.702,
    limit: float | None = None,
) -> AnyTensor:
    dispatch_args = (tensor,)
    for override in d.find_overrides(dispatch_args):
        result = override(tensor, alpha=alpha, limit=limit)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable
def to(tensor: AnyTensor, *args, **kwargs) -> AnyTensor:
    """See torch.Tensor.to"""
    ...


@to.trampoline
def _to_trampoline(d: SignatureDispatcher, tensor: AnyTensor, *args, **kwargs):
    dispatch_args = [tensor]
    for override in d.find_overrides(dispatch_args):
        result = override(tensor, *args, **kwargs)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable
def trace_tensor(key: str, *tensors: tuple[AnyTensor, ...]):
    """Trace tensor(s) in IREE runtime or in eager mode.

    You can add trace_tensor into your model wherever you want. It will insert a
    trace op into the IR. Then you can register a callback in the IREE runtime for
    custom handling of the trace command during execution. For example recording the
    tensor into a file. There is also a destination/sink for eager execution.

    The trace op will prevent fusion which will influence how the model is compiled.
    This may change the behavior of the program and cause a numerical issue to
    disappear if it was the result of op fusion.

    Example usage at sharktank/tests/ops/ops_test.py::TestTraceTensors.

    See:
    sharktank.utils.debugging.set_trace_tensor_callback
    sharktank.utils.debugging.trace_tensor_to_safetensors_callback
    sharktank.utils.debugging.flags.trace_path
    sharktank.utils.iree.make_hal_buffer_view_trace_default_callback
    sharktank.layers.BaseLayer.trace_tensor
    """
    ...


@trace_tensor.trampoline
def _trace_tensor_trampoline(
    d: SignatureDispatcher, key: str, *tensors: tuple[AnyTensor, ...]
):
    for override in d.find_overrides(tensors):
        result = override(key, *tensors)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable(is_trivially_replicable=False)
def barrier_on_logical_device(tensor: AnyTensor, ordinal: int) -> AnyTensor:
    """Transfer the tensor to a device with ordinal `ordinal`."""
    ...


@barrier_on_logical_device.trampoline
def _barrier_on_logical_device_trampoline(
    d: SignatureDispatcher, tensor: AnyTensor, ordinal: int
):
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor, ordinal)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable(is_trivially_replicable=False)
def transfer_to_logical_device(tensor: AnyTensor, ordinal: int) -> AnyTensor:
    """Transfer the tensor to a device with ordinal `ordinal`."""
    ...


@transfer_to_logical_device.trampoline
def _transfer_to_logical_device_trampoline(
    d: SignatureDispatcher, tensor: AnyTensor, ordinal: int
):
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor, ordinal)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def transpose(tensor: AnyTensor, dim0: int, dim1: int) -> AnyTensor:
    """See torch.transpose"""
    ...


@transpose.trampoline
def _transpose_trampoline(
    d: SignatureDispatcher, tensor: AnyTensor, dim0: int, dim1: int
) -> AnyTensor:
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor, dim0, dim1)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def unflatten(input: AnyTensor, dim: int, sizes: Tuple[int]) -> AnyTensor:
    """See torch.unflatten"""
    ...


@unflatten.trampoline
def _unflatten_trampoline(
    d: SignatureDispatcher, input: AnyTensor, dim: int, sizes: Tuple[int]
) -> AnyTensor:
    dispatch_args = (input,)
    for override in d.find_overrides(dispatch_args):
        result = override(input, dim, sizes)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable
def unpack(input: AnyTensor) -> QuantizedLayout:
    ...


@unpack.trampoline
def _unpack_trampoline(d: SignatureDispatcher, input: AnyTensor) -> QuantizedLayout:
    dispatch_args = (input,)
    for override in d.find_overrides(dispatch_args):
        result = override(input)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable
def unpack_qs(qs: AnyTensor, layout: BlockScaledPackedLayout) -> AnyTensor:
    """Return the unpacked unscaled/quantized values of a block scales packed layout."""
    ...


@unpack_qs.trampoline
def _qs_trampoline(
    d: SignatureDispatcher, qs: AnyTensor, layout: BlockScaledPackedLayout
) -> AnyTensor:
    dispatch_args = (
        qs,
        layout,
    )
    for override in d.find_overrides(dispatch_args):
        result = override(qs, layout)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable(is_trivially_replicable=False)
def unshard(tensor: AnyTensor) -> AnyTensor:
    """Return the tensor that has the same elements and shape, but is not sharded."""
    ...


@unshard.trampoline
def _unshard_trampoline(d: SignatureDispatcher, tensor: AnyTensor):
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def unsqueeze(tensor: AnyTensor, dim: int) -> AnyTensor:
    """See torch.unsqueeze"""
    ...


@unsqueeze.trampoline
def _unsqueeze_trampoline(
    d: SignatureDispatcher, tensor: AnyTensor, dim: int
) -> AnyTensor:
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor, dim)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def squeeze(tensor, dim: Optional[int]) -> AnyTensor:
    """See torch.squeeze"""
    ...


@squeeze.trampoline
def _squeeze_trampoline(
    d: SignatureDispatcher, tensor, dim: Optional[int]
) -> AnyTensor:
    tensors = (tensor,)
    for override in d.find_overrides(tensor):
        result = override(tensor, dim)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def sum(
    input,
    dim: Union[int, List[int]],
    keepdim: bool = False,
    *,
    dtype: torch.dtype = None,
) -> AnyTensor:
    """See torch.sum"""
    ...


@sum.trampoline
def _sum_trampoline(
    d: SignatureDispatcher,
    input,
    dim: int | List[int] | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype = None,
) -> AnyTensor:
    tensors = (input,)
    for override in d.find_overrides(tensors):
        result = override(input, dim=dim, keepdim=keepdim, dtype=dtype)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def topk(
    tensor,
    k: int,
    dim: int,
    largest: bool,
    sorted: bool,
    chunk_size: Optional[int] = None,
    use_linalgext_topk: bool = False,
) -> AnyTensor:
    """See torch.topk"""
    ...


@topk.trampoline
def _topk_trampoline(
    d: SignatureDispatcher,
    tensor,
    k: int,
    dim: int,
    largest: bool = True,
    sorted: bool = True,
    chunk_size: Optional[int] = None,
    use_linalgext_topk: bool = False,
) -> AnyTensor:
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        if isinstance(tensor, SplitPrimitiveTensor):
            result = override(
                tensor,
                k=k,
                dim=dim,
                largest=largest,
                sorted=sorted,
                use_linalgext_topk=use_linalgext_topk,
            )

        else:
            result = override(
                tensor,
                k=k,
                dim=dim,
                largest=largest,
                sorted=sorted,
                chunk_size=chunk_size,
                use_linalgext_topk=use_linalgext_topk,
            )
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def view(
    tensor: AnyTensor, shape: List[int] | None = None, dtype: torch.dtype | None = None
) -> AnyTensor:
    """See torch.Tensor.view"""
    ...


@view.trampoline
def _view_trampoline(
    d: SignatureDispatcher,
    tensor: AnyTensor,
    shape: List[int] | None = None,
    dtype: torch.dtype | None = None,
) -> AnyTensor:
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor, shape, dtype)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def view_as_complex(tensor: AnyTensor, shape: List[int]) -> AnyTensor:
    """See torch.Tensor.view_as_complex"""
    ...


@view_as_complex.trampoline
def _view_as_complex_trampoline(d: SignatureDispatcher, tensor: AnyTensor) -> AnyTensor:
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def view_as_real(tensor: AnyTensor, shape: List[int]) -> AnyTensor:
    """See torch.Tensor.view_as_complex"""
    ...


@view_as_real.trampoline
def _view_as_real_trampoline(d: SignatureDispatcher, tensor: AnyTensor) -> AnyTensor:
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def zeros_like(
    tensor: AnyTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool = False,
    memory_format: torch.memory_format = torch.preserve_format,
) -> AnyTensor:
    """See torch.zeros_like"""
    ...


@zeros_like.trampoline
def _zeros_like_trampoline(
    d: SignatureDispatcher,
    tensor: AnyTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool = False,
    memory_format: torch.memory_format = torch.preserve_format,
) -> AnyTensor:
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(
            tensor,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
            memory_format=memory_format,
        )
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)
