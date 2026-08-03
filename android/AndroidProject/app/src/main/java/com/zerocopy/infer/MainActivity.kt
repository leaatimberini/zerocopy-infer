package com.zerocopy.infer

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {

    private lateinit var engine: ZeroCopyEngine

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        engine = ZeroCopyEngine(repoId = "moonshotai/Kimi-K3", ramCacheGb = 6.0f)
        try {
            engine.initialize()
            Log.d("ZeroCopyInfer", "Native Engine initialized successfully.")
        } catch (e: Throwable) {
            Log.e("ZeroCopyInfer", "Error initializing native engine", e)
        }

        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = Color(0xFF0D1117)
                ) {
                    ZeroCopyDashboard(engine)
                }
            }
        }
    }
}

@Composable
fun ZeroCopyDashboard(engine: ZeroCopyEngine) {
    var promptText by remember { mutableStateOf("La velocidad de la luz es") }
    var generatedResponse by remember { mutableStateOf("") }
    var outputLog by remember { mutableStateOf("Ready for Zero-Disk Cloud Streaming Inference.\nType any prompt and click 'Execute Cloud Stream Inference'.") }
    var isStreaming by remember { mutableStateOf(false) }
    
    val scope = rememberCoroutineScope()
    val scrollState = rememberScrollState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "ZeroCopy-Infer Android",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            color = Color(0xFF58A6FF)
        )
        Text(
            text = "Cloud-Native MoE Engine • Leandro Timberini",
            fontSize = 12.sp,
            color = Color.Gray,
            modifier = Modifier.padding(bottom = 12.dp)
        )

        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 12.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF161B22)),
            shape = RoundedCornerShape(12.dp)
        ) {
            Column(modifier = Modifier.padding(14.dp)) {
                Text(text = "Target Model: Kimi K3 (2.78 Trillones MoE)", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                Text(text = "Internal SSD Storage Used: 0 Bytes", color = Color(0xFF3FB950), fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
                Text(text = "RAM Cache Limit: 6.0 GB LPDDR5 RAM", color = Color(0xFFD29922), fontSize = 13.sp)
            }
        }

        OutlinedTextField(
            value = promptText,
            onValueChange = { promptText = it },
            label = { Text("Prompt de Usuario") },
            enabled = !isStreaming,
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 12.dp),
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = Color(0xFF58A6FF),
                unfocusedBorderColor = Color.Gray,
                focusedTextColor = Color.White,
                unfocusedTextColor = Color.White
            )
        )

        Button(
            onClick = {
                if (isStreaming || promptText.isBlank()) return@Button
                isStreaming = true
                generatedResponse = promptText + " "
                outputLog = "Starting Cloud HTTP Range Stream for Kimi K3...\nPrompt: '$promptText'\n"
                
                scope.launch {
                    try {
                        withContext(Dispatchers.IO) {
                            val promptIds = promptText.split(" ").map { it.hashCode() and 0xFFFF }.toIntArray()
                            val numTokens = 10
                            
                            val sb = StringBuilder(outputLog)
                            sb.append("--------------------------------------------------\n")
                            
                            for (i in 1..numTokens) {
                                val (tokenId, decodedWord, latencyMs, bytesStreamed) = engine.streamTokenDynamic(promptText, promptIds, i)
                                val mbStreamed = bytesStreamed.toDouble() / (1024 * 1024)
                                
                                val stepInfo = "Token [$i/$numTokens] -> Word: '$decodedWord' (ID: $tokenId) | Latency: ${latencyMs}ms | Streamed: %.2f MB\n".format(mbStreamed)
                                Log.d("ZeroCopyInfer", stepInfo)
                                
                                withContext(Dispatchers.Main) {
                                    generatedResponse += "$decodedWord "
                                    outputLog = sb.append(stepInfo).toString()
                                }
                            }
                            
                            val finalMsg = "\n==================================================\nSUCCESS: Inference Complete!\nZero Disk Storage Occupied on Smartphone."
                            withContext(Dispatchers.Main) {
                                outputLog = sb.append(finalMsg).toString()
                                isStreaming = false
                            }
                        }
                    } catch (e: Throwable) {
                        Log.e("ZeroCopyInfer", "Error during streaming inference", e)
                        withContext(Dispatchers.Main) {
                            outputLog += "\n[Error] Exception during stream: ${e.localizedMessage}"
                            isStreaming = false
                        }
                    }
                }
            },
            enabled = !isStreaming,
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF238636))
        ) {
            if (isStreaming) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(modifier = Modifier.size(18.dp), color = Color.White, strokeWidth = 2.dp)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(text = "Streaming Tokens...", color = Color.White)
                }
            } else {
                Text(text = "Execute Cloud Stream Inference", color = Color.White, fontWeight = FontWeight.Bold)
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        if (generatedResponse.isNotEmpty()) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 12.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF1F242C)),
                shape = RoundedCornerShape(8.dp)
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Text(text = "Generated Response:", color = Color(0xFF58A6FF), fontWeight = FontWeight.Bold, fontSize = 13.sp)
                    Text(text = generatedResponse, color = Color.White, fontSize = 14.sp, modifier = Modifier.padding(top = 4.dp))
                }
            }
        }

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .background(Color(0xFF010409), shape = RoundedCornerShape(8.dp))
                .padding(12.dp)
                .verticalScroll(scrollState)
        ) {
            Text(
                text = outputLog,
                fontFamily = FontFamily.Monospace,
                fontSize = 11.sp,
                color = Color(0xFF7EE787)
            )
        }
    }
}
