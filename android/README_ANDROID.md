# ZeroCopy-Infer for Android (ARM64 C++23 NDK)

[![Target: Android ARM64](https://img.shields.io/badge/Target-Android%20ARM64-brightgreen.svg)]()
[![RAM Requirement](https://img.shields.io/badge/RAM%20Cache-4--6%20GB%20LRU-blue.svg)]()
[![Disk Storage Used](https://img.shields.io/badge/Internal%20Storage-0%20Bytes-success.svg)]()

**Authored and Created by**: Leandro Emanuel Timberini  
**Affiliation**: Investigador Independiente — Ituzaingó, Buenos Aires, Argentina  

---

## 🌟 Overview

`ZeroCopy-Infer for Android` brings **Zero-Disk Cloud-Native MoE & LLM Streaming Inference** directly to modern Android smartphones (such as Motorola Edge, Moto G series, Samsung Galaxy, and Google Pixel devices equipped with 12 GB RAM + RAM Boost).

Instead of requiring terabytes of internal UFS storage to store model weights, `ZeroCopy-Infer` streams individual layer and expert weight chunks in real time over Wi-Fi 6 or 5G directly into LPDDR5 RAM using **0 Bytes of internal smartphone storage**.

---

## 🛠️ Android Architecture

```
+------------------------------------+          HTTP Range Requests          +--------------------------------------+
|  Hugging Face Cloud CDN            | <===================================> |  Android Smartphone (ARM64)          |
|  Checkpoint MoE (1.56 TB)          |    (5G / Wi-Fi 6 Streaming of        |  (Motorola 12GB RAM + 6GB RAM Boost) |
|  [96 shards .safetensors en la nube]|     ~20 MB expert chunks into RAM)   |  [0 Bytes de Memoria Interna Usada] |
+------------------------------------+                                       +--------------------------------------+
                                                                                                │
                                                                                                ▼
                                                                                [libzerocopy_infer.so (C++23 / NDK)]
                                                                                [ZeroCopyEngine.kt (Kotlin JNI)]
```

- **Core Engine**: Compiled in C++23 using Android NDK (`clang++` for `arm64-v8a`) with **ARM NEON** vector instruction acceleration.
- **Storage Requirement**: **0 Bytes** (100% Zero-Disk RAM Ingest).
- **RAM Cache**: Configurable LRU ring buffer in host RAM (e.g. 4.0 GB to 6.0 GB RAM), leaving 6+ GB RAM free for Android OS.

---

## 🚀 Integration in Android Studio

1. Copy `CMakeLists.txt` and `zerocopy_infer_jni.cpp` to your Android project `app/src/main/cpp/`.
2. Add CMake build path to `app/build.gradle.kts`:

```kotlin
android {
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    ndkVersion = "25.2.9519653"
}
```

3. Include `ZeroCopyEngine.kt` in your Kotlin app:

```kotlin
val engine = ZeroCopyEngine(repoId = "moonshotai/Kimi-K3", ramCacheGb = 6.0f)
engine.initialize()

// Stream forward token pass
val (tokenId, latencyMs, bytesStreamed) = engine.streamToken(intArrayOf(1008, 10484, 318))
println("Generated Token: $tokenId in ${latencyMs}ms | Streamed: ${bytesStreamed / (1024*1024)} MB")
```

---

## 📜 License

Distributed under the MIT License. See [LICENSE](../LICENSE) for details.

**Author**: Leandro Emanuel Timberini  
**Affiliation**: Investigador Independiente — Ituzaingó, Buenos Aires, Argentina
