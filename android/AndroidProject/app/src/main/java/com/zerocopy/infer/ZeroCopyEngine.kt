package com.zerocopy.infer

/**
 * ZeroCopyEngine.kt
 * =================
 * Android Kotlin API wrapper for ZeroCopy-Infer native C++23 ARM64 engine.
 * Authored by Leandro Emanuel Timberini (Investigador Independiente — Ituzaingó, Buenos Aires, Argentina).
 *
 * Executes zero-disk cloud streaming MoE inference on Android smartphones
 * (e.g., Motorola Edge / Moto G series with 12 GB RAM + 6 GB RAM Boost)
 * using 0 Bytes of local storage.
 */
class ZeroCopyEngine(
    val repoId: String = "moonshotai/Kimi-K3",
    val ramCacheGb: Float = 6.0f
) {
    companion object {
        init {
            System.loadLibrary("zerocopy_infer")
        }
    }

    private external fun nativeGetVersion(): String
    private external fun nativeInitEngine(repoId: String, ramCacheGb: Float): Boolean
    private external fun nativeStreamToken(promptIds: IntArray): LongArray

    fun getEngineVersion(): String {
        return nativeGetVersion()
    }

    fun initialize(): Boolean {
        return nativeInitEngine(repoId, ramCacheGb)
    }

    /**
     * Streams a single forward token pass using cloud HTTP Range Requests.
     * Returns Triple(tokenId, latencyMs, bytesStreamed)
     */
    fun streamToken(promptIds: IntArray): Triple<Long, Long, Long> {
        val res = nativeStreamToken(promptIds)
        return Triple(res[0], res[1], res[2])
    }
}
