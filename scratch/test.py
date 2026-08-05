import numpy as np

FP4_E2M1_TABLE = np.array([
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=np.float32)

def dequantize_mxfp4_orig(weight_packed, weight_scale, block_size=32):
    low_nibble = weight_packed & 0x0F
    high_nibble = (weight_packed >> 4) & 0x0F
    shape_N = weight_packed.shape[0]
    fp4_indices = np.stack([low_nibble, high_nibble], axis=-1).reshape(shape_N, -1)
    unscaled_weights = FP4_E2M1_TABLE[fp4_indices]
    scale_floats = weight_scale.astype(np.float32)
    scales = np.power(2.0, scale_floats - 127.0)
    scale_expanded = np.repeat(scales, block_size, axis=-1)[:, :unscaled_weights.shape[1]]
    return unscaled_weights * scale_expanded

def dequantize_mxfp4_opt(weight_packed, weight_scale, block_size=32):
    shape_N, half_K = weight_packed.shape
    fp4_indices = np.empty((shape_N, half_K * 2), dtype=np.uint8)
    fp4_indices[:, 0::2] = weight_packed & 0x0F
    fp4_indices[:, 1::2] = weight_packed >> 4
    unscaled_weights = FP4_E2M1_TABLE[fp4_indices]
    scales = np.ldexp(1.0, weight_scale.astype(np.int32) - 127).astype(np.float32)
    scales = scales[:, :, np.newaxis]
    unscaled_weights = unscaled_weights.reshape(shape_N, -1, block_size)
    return (unscaled_weights * scales).reshape(shape_N, -1)

wp = np.random.randint(0, 256, (10, 64), dtype=np.uint8)
ws = np.random.randint(0, 256, (10, 4), dtype=np.uint8)

res1 = dequantize_mxfp4_orig(wp, ws)
res2 = dequantize_mxfp4_opt(wp, ws)

print(np.allclose(res1, res2))
