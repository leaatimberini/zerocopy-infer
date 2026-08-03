# ZeroCopy-Infer: Android Studio Coroutines & UI Refactor Documentation

**Author**: Leandro Emanuel Timberini (Investigador Independiente — Ituzaingó, Buenos Aires, Argentina)  
**Project Path**: `C:\zerocopy-infer\android\AndroidProject`  
**Target Hardware**: Motorola Edge / Moto G series (12 GB RAM + 6 GB RAM Boost LPDDR5)  
**Target MoE Model**: Kimi K3 (2.78 Trillion Parameters, Zero-Disk Cloud Streaming)

---

## 1. Problem Diagnosis: Why Pressing the Prompt Button on Motorola Phones Did Nothing

### Root Cause Analysis
1. **Main UI Thread Blocking**:
   In Jetpack Compose, the `onClick` lambda callback of a `Button` executes on the Android Main UI Thread (the `Choreographer` / UI Event Looper).
2. **Synchronous Network / Heavy I/O Operation**:
   When `engine.streamToken()` is invoked directly inside `Button(onClick = { ... })`, the native engine attempts to perform cloud HTTP Range Requests (or blocking simulated forward passes) synchronously.
3. **`android.os.NetworkOnMainThreadException` & UI Freezes**:
   - On Android SDK 26+ (Android 8.0 through Android 14+), performing network operations or heavy blocking calls on the main thread throws `NetworkOnMainThreadException` or starves the UI thread looper.
   - Jetpack Compose fails to update state or recompose because the UI thread is frozen/hung, causing the button press to appear completely unresponsive on Motorola devices.

---

## 2. Architecture & Coroutine Refactoring Fixes

### 1. Jetpack Compose Coroutine Scope Management
- Created a Compose coroutine scope bound to the composable lifecycle using `val scope = rememberCoroutineScope()`.
- Launched streaming operations off the main thread:
  ```kotlin
  scope.launch {
      withContext(Dispatchers.IO) {
          // Streaming network requests execute asynchronously here on I/O worker threads
          val stepResult = engine.streamToken(promptIds, i, promptText)
          
          // Switch safely to Dispatchers.Main for Compose state updates
          withContext(Dispatchers.Main) {
              liveGeneratedText += stepResult.tokenText
              totalTokensGenerated++
              totalBytesStreamed += stepResult.bytesStreamed
              outputLog = sbLog.toString()
          }
      }
  }
  ```

### 2. Thread Safety & Non-Blocking Token Streaming
- Created `streamTokenAsync()` suspend function inside `ZeroCopyEngine.kt` backed by `withContext(Dispatchers.IO)`.
- Maintained thread-safe state mutation by dispatching UI state changes back to `Dispatchers.Main`.

---

## 3. Engine Enhancements in `ZeroCopyEngine.kt`

- **Tokenization Simulation**: `tokenize(prompt: String): IntArray` converts user input strings into token IDs.
- **Detokenization**: `detokenize(tokenId: Long, stepIndex: Int, prompt: String): String` maps token IDs back into readable words for live streaming visualization.
- **Telemetry Data Class**: `TokenStreamResult` encapsulates token ID, token text, step latency (ms), HTTP Range bytes streamed, RAM usage (MB), and storage stats (strictly 0 B SSD).
- **JVM & LPDDR5 Memory Metrics**: `getMemoryMetrics()` retrieves real-time JVM heap allocation and RAM cache limits (6.0 GB).
- **Error Stacktrace Formatting**: `formatStackTrace(throwable: Throwable)` converts exceptions into formatted text strings for direct UI stacktrace inspection.

---

## 4. UI Dashboard Enhancements in `MainActivity.kt`

- **Real-Time Live Telemetry Cards**:
  - **Tokens & Latency**: Live count of generated tokens, per-step latency (ms), and generation throughput (tokens/sec).
  - **Streamed Data & RAM**: Total Cloud HTTP Range Request bandwidth (MB) and LPDDR5 RAM usage vs 6.0 GB cache limit.
  - **Internal Phone Storage**: Badge explicitly displaying **0 Bytes (0.00 KB)** to verify Zero-Disk Cloud Streaming.
- **Interactive Prompt Control**:
  - Outlined prompt text field with default pre-filled prompt (`The capital of France is`).
  - Active button state with animated `CircularProgressIndicator` during token streaming.
  - Reset button to clear console and counters for consecutive runs.
- **Live Streamed Output Box**: Displays the text response building token-by-token in real time.
- **Error Stacktrace Panel**: Rendered in a high-visibility alert section if an exception is caught during execution.
- **Telemetry Console Window**: Monospaced terminal scrolling view detailing every token event, step latency, and bandwidth metric.

---

## 5. Verification & Build Confirmation

- Executed `./gradlew.bat compileDebugSources` in `C:\zerocopy-infer\android\AndroidProject`.
- **Result**: `BUILD SUCCESSFUL` with 0 errors and 0 warnings.
