import os

import torch
import torch.nn as nn
from torch.nn import functional as F


def get_fp_maxval(bits=8, mantissa_bit=3, sign_bits=1):
    """Calculate the maximum representable value for a custom floating-point format
    
    This function computes the maximum value that can be represented in a
    floating-point format with specified bit configuration.
    
    Args:
        bits (int): Total number of bits for the floating-point format
        mantissa_bit (int): Number of bits for the mantissa (fractional part)
        sign_bits (int): Number of bits for the sign
        
    Returns:
        float: Maximum representable value in the specified format
    """
    _bits = torch.tensor(bits)
    _mantissa_bit = torch.tensor(mantissa_bit)
    _sign_bits = torch.tensor(sign_bits)
    
    # Calculate exponent bits: total bits - sign bits - mantissa bits
    M = torch.clamp(torch.round(_mantissa_bit), 1, _bits - _sign_bits)
    E = _bits - _sign_bits - M
    
    # Calculate bias for exponent (standard IEEE 754 bias)
    bias = 2 ** (E - 1) - 1
    
    # Calculate maximum mantissa value (all bits set to 1)
    mantissa = 1
    for i in range(mantissa_bit - 1):
        mantissa += 1 / (2 ** (i+1))
    
    # Calculate maximum value: mantissa * 2^(max_exponent - bias)
    maxval = mantissa * 2 ** (2**E - 1 - bias)
    return maxval


def quantize_to_fp8(x, bits=8, mantissa_bit=3, sign_bits=1):
    """Quantize input tensor to custom floating-point format (default E4M3)
    
    This function performs quantization to a custom floating-point format,
    typically E4M3 (4 exponent bits, 3 mantissa bits) for FP8.
    
    Args:
        x (torch.Tensor): Input tensor to quantize
        bits (int): Total number of bits for the format
        mantissa_bit (int): Number of mantissa bits
        sign_bits (int): Number of sign bits
        
    Returns:
        tuple: (quantized_tensor, log_scales) where log_scales are the quantization scales
    """
    bits = torch.tensor(bits)
    mantissa_bit = torch.tensor(mantissa_bit)
    sign_bits = torch.tensor(sign_bits)
    
    # Calculate format parameters
    M = torch.clamp(torch.round(mantissa_bit), 1, bits - sign_bits)
    E = bits - sign_bits - M
    bias = 2 ** (E - 1) - 1
    
    # Calculate maximum mantissa value
    mantissa = 1
    for i in range(mantissa_bit - 1):
        mantissa += 1 / (2 ** (i+1))
    
    # Calculate dynamic range
    maxval = mantissa * 2 ** (2**E - 1 - bias)
    minval = - maxval
    minval = - maxval if sign_bits == 1 else torch.zeros_like(maxval)
    
    # Clamp input to valid range
    input_clamp = torch.min(torch.max(x, minval), maxval)
    
    # Calculate quantization scales based on magnitude
    log_scales = torch.clamp(
        (torch.floor(torch.log2(torch.abs(input_clamp)) + bias)).detach(), 1.0)
    log_scales = 2.0 ** (log_scales - M - bias.type(x.dtype))
    
    # Perform quantization and dequantization (round-trip)
    qdq_out = torch.round(input_clamp / log_scales) * log_scales
    return qdq_out, log_scales


def fp8_tensor_quant(x, scale, bits=8, mantissa_bit=3, sign_bits=1):
    """Quantize tensor with pre-computed scale to FP8 format
    
    This function applies scaling and quantization to a tensor using
    a pre-computed scale factor.
    
    Args:
        x (torch.Tensor): Input tensor
        scale (torch.Tensor): Pre-computed scale factor
        bits (int): Total bits for FP format
        mantissa_bit (int): Mantissa bits
        sign_bits (int): Sign bits
        
    Returns:
        tuple: (quantized_tensor, scale, log_scales)
    """
    # Expand scale dimensions to match input tensor
    for i in range(len(x.shape) - 1):
        scale = scale.unsqueeze(-1)
    
    # Apply scaling and quantization
    new_x = x / scale
    quant_dequant_x, log_scales = quantize_to_fp8(
        new_x, bits=bits, mantissa_bit=mantissa_bit, sign_bits=sign_bits)
    return quant_dequant_x, scale, log_scales


def fp8_activation_dequant(qdq_out, scale, dtype):
    """Dequantize FP8 tensor back to original precision
    
    This function converts quantized FP8 tensors back to the original
    data type by applying the scale factor.
    
    Args:
        qdq_out (torch.Tensor): Quantized tensor
        scale (torch.Tensor): Scale factor for dequantization
        dtype: Target data type for dequantization
        
    Returns:
        torch.Tensor: Dequantized tensor in target dtype
    """
    qdq_out = qdq_out.type(dtype)
    quant_dequant_x = qdq_out * scale.to(dtype)
    return quant_dequant_x


def fp8_linear_forward(cls, original_dtype, input):
    """Forward pass for FP8-optimized linear layers
    
    This function replaces the standard linear forward pass with an FP8-optimized
    version that handles quantization and dequantization automatically.
    
    Args:
        cls: Linear layer instance
        original_dtype: Original data type of the layer
        input (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Output of the linear transformation
    """
    weight_dtype = cls.weight.dtype
    
    # ===== Weight quantization section =====
    if cls.weight.dtype != torch.float8_e4m3fn:
        # Convert weights to FP8 if not already in FP8 format
        maxval = get_fp_maxval()
        scale = torch.max(torch.abs(cls.weight.flatten())) / maxval
        linear_weight, scale, log_scales = fp8_tensor_quant(cls.weight, scale)
        linear_weight = linear_weight.to(torch.float8_e4m3fn)
        weight_dtype = linear_weight.dtype
    else:
        # Use pre-computed scale for already FP8 weights
        scale = cls.fp8_scale.to(cls.weight.device)
        linear_weight = cls.weight
    # ===== End weight quantization =====

    # Perform FP8-optimized linear transformation
    if weight_dtype == torch.float8_e4m3fn and cls.weight.sum() != 0:
        if True or len(input.shape) == 3:
            # Dequantize weights to original precision for computation
            cls_dequant = fp8_activation_dequant(
                linear_weight, scale, original_dtype)
            
            # Perform linear transformation with bias if available
            if cls.bias != None:
                output = F.linear(input, cls_dequant, cls.bias)
            else:
                output = F.linear(input, cls_dequant)
            return output
        else:
            # Fallback to original forward for non-3D inputs
            return cls.original_forward(input.to(original_dtype))
    else:
        # Fallback to original forward for non-FP8 weights or zero weights
        return cls.original_forward(input)


def convert_fp8_linear(module, dit_weight_path, original_dtype, params_to_keep={}):
    """Convert linear layers in a module to use FP8 optimization
    
    This function modifies linear layers in a module to use FP8 quantization
    for improved memory efficiency and potentially faster computation.
    
    Args:
        module: PyTorch module containing linear layers
        dit_weight_path (str): Path to the model weights file
        original_dtype: Original data type to maintain compatibility
        params_to_keep (dict): Additional parameters to preserve
        
    Raises:
        ValueError: If FP8 mapping file is not found
    """
    # Enable FP8 matrix multiplication flag
    setattr(module, "fp8_matmul_enabled", True)

    # Load FP8 quantization mapping from file
    fp8_map_path = dit_weight_path.replace('.pt', '_map.pt')
    if os.path.exists(fp8_map_path):
        fp8_map = torch.load(
            fp8_map_path, map_location=lambda storage, loc: storage)
    else:
        raise ValueError(f"Invalid fp8_map path: {fp8_map_path}.")

    # Convert eligible linear layers to FP8
    fp8_layers = []
    for key, layer in module.named_modules():
        # Target specific linear layers in double_blocks and single_blocks
        if isinstance(layer, nn.Linear) and ('double_blocks' in key or 'single_blocks' in key):
            fp8_layers.append(key)
            
            # Store original forward method
            original_forward = layer.forward
            
            # Convert weights to FP8 format
            layer.weight = torch.nn.Parameter(
                layer.weight.to(torch.float8_e4m3fn))
            
            # Set FP8 scale from mapping file
            setattr(layer, "fp8_scale", fp8_map[key].to(dtype=original_dtype))
            
            # Store original forward method for fallback
            setattr(layer, "original_forward", original_forward)
            
            # Replace forward method with FP8-optimized version
            setattr(layer, "forward", lambda input,
                    m=layer: fp8_linear_forward(m, original_dtype, input))
