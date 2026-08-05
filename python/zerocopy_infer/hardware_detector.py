import os
import platform
import multiprocessing
import subprocess

class HardwareDetector:
    @classmethod
    def get_system_info(cls):
        system = platform.system()
        if system == "Linux" and hasattr(platform, "android_ver"):
            system = "Android"
        elif "termux" in os.environ.get("PREFIX", "").lower():
            system = "Android Termux"
        
        arch = platform.machine().lower()
        return system, arch

    @classmethod
    def get_cpu_cores(cls):
        try:
            return multiprocessing.cpu_count()
        except NotImplementedError:
            return 4

    @classmethod
    def get_memory_info(cls):
        total_ram = 0
        total_swap = 0
        system, _ = cls.get_system_info()
        
        try:
            if system in ["Linux", "Android Termux"]:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            total_ram = int(line.split()[1]) * 1024
                        elif line.startswith("SwapTotal:"):
                            total_swap = int(line.split()[1]) * 1024
            elif system == "Windows":
                # Use wmic to get memory on Windows
                ram_out = subprocess.check_output('wmic computersystem get TotalPhysicalMemory', shell=True).decode()
                total_ram = int(ram_out.split()[1].strip())
                # Just placeholder for swap on Windows
                total_swap = 0 
        except Exception:
            pass
            
        return total_ram, total_swap

    @classmethod
    def get_simd_features(cls):
        system, arch = cls.get_system_info()
        features = []
        
        if arch in ["x86_64", "amd64", "x86"]:
            try:
                if system in ["Linux", "Android Termux"]:
                    with open("/proc/cpuinfo", "r") as f:
                        content = f.read()
                        if "avx512" in content:
                            features.append("AVX-512")
                        if "avx2" in content:
                            features.append("AVX2")
                        if "avx " in content:
                            features.append("AVX")
                elif system == "Windows":
                    # Simple fallback for Windows x86_64, assuming AVX2 on modern CPUs
                    features.append("AVX2")
            except Exception:
                pass
        elif arch in ["aarch64", "arm64"]:
            features.append("NEON")
            
        return features

    @classmethod
    def detect(cls):
        system, arch = cls.get_system_info()
        cpu_cores = cls.get_cpu_cores()
        total_ram, total_swap = cls.get_memory_info()
        simd = cls.get_simd_features()
        
        return {
            "os": system,
            "architecture": arch,
            "cpu_cores": cpu_cores,
            "threads": cpu_cores, # Typically thread count maps to core count here
            "simd": simd,
            "total_ram": total_ram,
            "total_swap": total_swap
        }

if __name__ == "__main__":
    hw_info = HardwareDetector.detect()
    print("Detected Hardware:")
    for k, v in hw_info.items():
        if "ram" in k or "swap" in k:
            print(f"  {k}: {v / (1024**3):.2f} GB")
        else:
            print(f"  {k}: {v}")
