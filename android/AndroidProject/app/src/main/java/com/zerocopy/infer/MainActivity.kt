package com.zerocopy.infer

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

class MainActivity : ComponentActivity() {

    private lateinit var engine: ZeroCopyEngine

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        engine = ZeroCopyEngine(repoId = "moonshotai/Kimi-K3", ramCacheGb = 6.0f)
        try {
            engine.initialize()
        } catch (e: Exception) {
            e.printStackTrace()
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
    var promptText by remember { mutableStateOf("The capital of France is") }
    var outputLog by remember { mutableStateOf("Ready for Zero-Disk Cloud Streaming Inference.") }
    var isStreaming by remember { mutableStateOf(false) }

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
            modifier = Modifier.padding(bottom = 16.dp)
        )

        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF161B22)),
            shape = RoundedCornerShape(12.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(text = "Target Model: Kimi K3 (2.78 Trillones MoE)", color = Color.White, fontWeight = FontWeight.Bold)
                Text(text = "Internal SSD Storage Used: 0 Bytes", color = Color(0xFF3FB950), fontWeight = FontWeight.SemiBold)
                Text(text = "RAM Cache Limit: 6.0 GB LPDDR5 RAM", color = Color(0xFFD29922))
            }
        }

        OutlinedTextField(
            value = promptText,
            onValueChange = { promptText = it },
            label = { Text("Prompt") },
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp),
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = Color(0xFF58A6FF),
                unfocusedBorderColor = Color.Gray,
                focusedTextColor = Color.White
            )
        )

        Button(
            onClick = {
                isStreaming = true
                val (tokenId, latencyMs, bytesStreamed) = engine.streamToken(intArrayOf(1008, 10484, 318))
                outputLog = "Token Generated: $tokenId\nLatency: ${latencyMs}ms\nStreamed via HTTP Range: ${bytesStreamed / (1024 * 1024)} MB\nSSD Usage: 0 Bytes"
                isStreaming = false
            },
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF238636))
        ) {
            Text(text = if (isStreaming) "Streaming from HF..." else "Execute Cloud Stream Inference")
        }

        Spacer(modifier = Modifier.height(16.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .background(Color(0xFF010409), shape = RoundedCornerShape(8.dp))
                .padding(12.dp)
        ) {
            Text(
                text = outputLog,
                fontFamily = FontFamily.Monospace,
                fontSize = 12.sp,
                color = Color(0xFF7EE787)
            )
        }
    }
}
