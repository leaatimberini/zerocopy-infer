import numpy as np
from .hardware_detector import HardwareDetector

from .mxfp4_dequant import dequantize_mxfp4 as fast_dequantize_mxfp4

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
        return np.matmul(a, b)
            
    def dot_product(self, a, b):
        return np.dot(a, b)
            
    def dequantize_mxfp4(self, tensor, scales):
        return fast_dequantize_mxfp4(tensor, scales)
        
global_kernels = OptimizedKernels()

def dispatch_matmul(a, b):
    return global_kernels.matmul(a, b)

def dispatch_dot(a, b):
    return global_kernels.dot_product(a, b)
    
def dispatch_mxfp4_dequant(tensor, scales):
    return global_kernels.dequantize_mxfp4(tensor, scales)
