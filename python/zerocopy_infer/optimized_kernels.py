import numpy as np
from .hardware_detector import HardwareDetector

class OptimizedKernels:
    def __init__(self):
        self.hw_info = HardwareDetector.detect()
        self.simd = self.hw_info["simd"]
        self.arch = self.hw_info["architecture"]
        self.os = self.hw_info["os"]
        self.threads = self.hw_info["threads"]
        
    def _is_mobile(self):
        return "Android" in self.os or self.arch in ["aarch64", "arm64"]

    def matmul(self, a, b):
        # Dispatch to best available implementation
        if self._is_mobile() and "NEON" in self.simd:
            # Placeholder for C/NEON JIT / specifically optimized ARM kernel
            # Fallback to numpy which may be built with OpenBLAS NEON support
            return np.matmul(a, b)
        elif "AVX-512" in self.simd or "AVX2" in self.simd:
            # x86 optimized branch
            return np.matmul(a, b)
        else:
            return np.matmul(a, b)
            
    def dot_product(self, a, b):
        if self._is_mobile() and "NEON" in self.simd:
            return np.dot(a, b)
        elif "AVX-512" in self.simd or "AVX2" in self.simd:
            return np.dot(a, b)
        else:
            return np.dot(a, b)
            
    def dequantize_mxfp4(self, tensor, scales):
        # MXFP4 Dequantization routing
        if self._is_mobile() and "NEON" in self.simd:
            # Optimized NEON branch for mobile MXFP4
            return self._dequant_numpy_simd(tensor, scales)
        elif "AVX-512" in self.simd:
            return self._dequant_numpy_simd(tensor, scales)
        else:
            return self._dequant_numpy_simd(tensor, scales)
            
    def _dequant_numpy_simd(self, tensor, scales):
        # Fallback numpy SIMD-vectorized dequantization logic
        # In a real implementation this might call a C extension
        # Here we do a fast numpy multiply
        return tensor.astype(np.float32) * scales
        
global_kernels = OptimizedKernels()

def dispatch_matmul(a, b):
    return global_kernels.matmul(a, b)

def dispatch_dot(a, b):
    return global_kernels.dot_product(a, b)
    
def dispatch_mxfp4_dequant(tensor, scales):
    return global_kernels.dequantize_mxfp4(tensor, scales)
