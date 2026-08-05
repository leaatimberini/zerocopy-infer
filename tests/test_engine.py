"""
ZeroCopy-Infer: Automated Test & Verification Suite
===================================================
Tests all 6 model architecture handlers, hardware auto-detector, SIMD kernel dispatcher, 
and tokenizer prompt templates.
"""

import sys
import unittest
import numpy as np
from typing import Dict, Any

from python.zerocopy_infer.hardware_detector import detect_hardware, HardwareDetector
from python.zerocopy_infer.optimized_kernels import dispatch_matmul, dispatch_mxfp4_dequant
from python.zerocopy_infer.model_architectures import (
    get_architecture_handler,
    Gemma4ArchitectureHandler,
    KimiK3ArchitectureHandler,
    XiaomiMiMoArchitectureHandler,
    DeepSeekV3ArchitectureHandler,
    Qwen25ArchitectureHandler,
    MixtralArchitectureHandler,
)
from python.zerocopy_infer.tokenizer import UniversalHFTokenizer


class TestHardwareDetector(unittest.TestCase):
    def test_detect_hardware(self):
        hw = detect_hardware()
        self.assertIn("arch", hw)
        self.assertIn("system", hw)
        self.assertIn("cpu_count", hw)
        self.assertIn("simd_extension", hw)
        self.assertGreater(hw["cpu_count"], 0)


class TestOptimizedKernels(unittest.TestCase):
    def test_dispatch_matmul(self):
        W = np.random.randn(32, 64).astype(np.float32)
        x = np.random.randn(64).astype(np.float32)
        res = dispatch_matmul(W, x)
        expected = W @ x
        np.testing.assert_allclose(res, expected, rtol=1e-4, atol=1e-4)

    def test_dispatch_mxfp4_dequant(self):
        packed = np.random.randint(0, 255, (4, 16), dtype=np.uint8)
        scale = np.random.randint(120, 135, (4, 1), dtype=np.uint8)
        res = dispatch_mxfp4_dequant(packed, scale)
        self.assertEqual(res.shape, (4, 32))


class TestModelArchitectureHandlers(unittest.TestCase):
    def test_gemma4_handler(self):
        handler = get_architecture_handler("google/gemma-4-26B-A4B-it", {"model_type": "gemma4"})
        self.assertIsInstance(handler, Gemma4ArchitectureHandler)
        tensor_names = handler.get_tensor_name("q_proj", 0)
        self.assertIn("model.language_model.layers.0.self_attn.q_proj.weight", tensor_names)
        
        # Test softcapping
        logits = np.array([10.0, 50.0, -100.0], dtype=np.float32)
        softcapped = handler.apply_logit_softcapping(logits)
        self.assertLessEqual(np.max(softcapped), 30.0)

    def test_kimi_k3_handler(self):
        handler = get_architecture_handler("moonshotai/Kimi-K3", {"model_type": "kimi_k3"})
        self.assertIsInstance(handler, KimiK3ArchitectureHandler)
        tensor_names = handler.get_tensor_name("moe_router", 0)
        self.assertIn("model.layers.0.block_sparse_moe.gate.weight", tensor_names)

    def test_mimo_handler(self):
        handler = get_architecture_handler("XiaomiMiMo/MiMo-V2.5-Pro", {"model_type": "mimo"})
        self.assertIsInstance(handler, XiaomiMiMoArchitectureHandler)

    def test_deepseek_v3_handler(self):
        handler = get_architecture_handler("deepseek-ai/DeepSeek-V3", {"model_type": "deepseek_v3"})
        self.assertIsInstance(handler, DeepSeekV3ArchitectureHandler)

    def test_qwen25_handler(self):
        handler = get_architecture_handler("Qwen/Qwen2.5-1.5B-Instruct", {"model_type": "qwen2"})
        self.assertIsInstance(handler, Qwen25ArchitectureHandler)

    def test_mixtral_handler(self):
        handler = get_architecture_handler("mistralai/Mixtral-8x7B-Instruct-v0.1", {})
        self.assertIsInstance(handler, MixtralArchitectureHandler)


class TestUniversalTokenizer(unittest.TestCase):
    def test_chat_prompt_rendering(self):
        tok = UniversalHFTokenizer(repo_id="google/gemma-4-26B-A4B-it")
        prompt = tok.render_chat_prompt("hola", "google/gemma-4-26B-A4B-it")
        self.assertTrue("<|turn>user\nhola<turn|>" in prompt)

        messages = [{"role": "user", "content": "hola"}, {"role": "assistant", "content": "buenas"}]
        prompt_multi = tok.render_chat_prompt(messages, "google/gemma-4-26B-A4B-it")
        self.assertTrue("<|turn>model\nbuenas<turn|>" in prompt_multi)


class TestZeroCopyContextManager(unittest.TestCase):
    def test_kv_cache_sliding_window(self):
        from python.zerocopy_infer.context_manager import ZeroCopyContextManager
        ctx = ZeroCopyContextManager(max_context_length=100, sliding_window=10)
        for i in range(25):
            k = np.random.randn(64).astype(np.float32)
            v = np.random.randn(64).astype(np.float32)
            k_seq, v_seq = ctx.update_kv_cache(layer_idx=0, k_new=k, v_new=v)
        self.assertEqual(k_seq.shape[0], 10)
        self.assertEqual(v_seq.shape[0], 10)
        self.assertEqual(ctx.cached_layers_count, 1)


class TestWebDashboard(unittest.TestCase):
    def test_dashboard_import(self):
        from examples.web_dashboard import TelemetryHandler
        self.assertIsNotNone(TelemetryHandler)


class TestBenchmark(unittest.TestCase):
    def test_benchmark_import(self):
        from examples.benchmark import run_benchmark
        self.assertIsNotNone(run_benchmark)


class TestConfigWizard(unittest.TestCase):
    def test_config_wizard_import(self):
        from examples.config_wizard import run_config_wizard
        self.assertIsNotNone(run_config_wizard)


class TestMemoryPressureGuard(unittest.TestCase):
    def test_memory_guard(self):
        from python.zerocopy_infer.memory_guard import MemoryPressureGuard
        purged = [False]
        def dummy_purge():
            purged[0] = True
            return 1024
        guard = MemoryPressureGuard(target_max_ram_ratio=0.01, purge_callback=dummy_purge)
        bytes_purged = guard.enforce_safety()
        self.assertTrue(purged[0])
        self.assertEqual(bytes_purged, 1024)


class TestOpenAIApiServer(unittest.TestCase):
    def test_api_server_import(self):
        from examples.api_server import OpenAIApiHandler
        self.assertIsNotNone(OpenAIApiHandler)


class TestZeroCopyClient(unittest.TestCase):
    def test_client_import(self):
        from python.zerocopy_infer.client import ZeroCopyClient
        self.assertIsNotNone(ZeroCopyClient)


if __name__ == "__main__":
    unittest.main()
