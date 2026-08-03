package com.zerocopy.infer

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
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
                    ZeroCopyChatInterface(engine)
                }
            }
        }
    }
}

@Composable
fun ZeroCopyChatInterface(engine: ZeroCopyEngine) {
    var inputText by remember { mutableStateOf("") }
    val chatMessages = remember { mutableStateListOf<ChatMessage>() }
    var isStreaming by remember { mutableStateOf(false) }
    var activeThinkingText by remember { mutableStateOf("") }
    var activeImageBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var activeAssistantMessage by remember { mutableStateOf("") }
    var isTokenizerReady by remember { mutableStateOf(false) }
    
    val listState = rememberLazyListState()
    val scope = rememberCoroutineScope()

    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            engine.loadRemoteKimiTokenizerOnPhone()
        }
        isTokenizerReady = true
    }

    LaunchedEffect(chatMessages.size, activeAssistantMessage, activeThinkingText, activeImageBitmap) {
        if (chatMessages.isNotEmpty() || activeAssistantMessage.isNotEmpty() || activeThinkingText.isNotEmpty() || activeImageBitmap != null) {
            listState.animateScrollToItem((chatMessages.size + if (isStreaming) 1 else 0).coerceAtLeast(0))
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF0D1117))
    ) {
        // Header Bar
        Surface(
            modifier = Modifier.fillMaxWidth(),
            color = Color(0xFF161B22),
            shadowElevation = 4.dp
        ) {
            Column(modifier = Modifier.padding(14.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = "Kimi-K3 Real Inferencia",
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF58A6FF)
                    )
                    Spacer(modifier = Modifier.weight(1f))
                    Surface(
                        color = Color(0xFF238636),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Text(
                            text = "Real Zero-Copy MoE",
                            color = Color.White,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
                        )
                    }
                }
                Text(
                    text = "Inferencia Real Cloud SSE Stream • Hugging Face Zero-Copy • Safetensors Direct",
                    fontSize = 11.sp,
                    color = Color.Gray,
                    modifier = Modifier.padding(top = 2.dp)
                )
            }
        }

        // Chat Messages Area
        LazyColumn(
            state = listState,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            if (chatMessages.isEmpty() && !isStreaming) {
                item {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 40.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = if (isTokenizerReady) 
                                "¡Hola! Escribe tu consulta para realizar inferencia real streaming con Kimi-K3."
                            else 
                                "Cargando tokenizador oficial Kimi-K3 TikToken en la RAM de tu Motorola...",
                            color = Color.Gray,
                            fontSize = 13.sp
                        )
                    }
                }
            }

            items(chatMessages) { msg ->
                ChatBubble(msg)
            }

            if (isStreaming) {
                item {
                    ChatBubble(
                        ChatMessage(
                            sender = "assistant",
                            text = activeAssistantMessage,
                            thinkingText = activeThinkingText,
                            imageBitmap = activeImageBitmap
                        )
                    )
                }
            }
        }

        // Input Bar
        Surface(
            modifier = Modifier.fillMaxWidth(),
            color = Color(0xFF161B22),
            shadowElevation = 8.dp
        ) {
            Row(
                modifier = Modifier
                    .padding(8.dp)
                    .fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value = inputText,
                    onValueChange = { inputText = it },
                    placeholder = { Text("Escribe tu consulta...", color = Color.Gray) },
                    enabled = !isStreaming,
                    modifier = Modifier.weight(1f),
                    shape = RoundedCornerShape(24.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = Color(0xFF58A6FF),
                        unfocusedBorderColor = Color(0xFF30363D),
                        focusedTextColor = Color.White,
                        unfocusedTextColor = Color.White,
                        focusedContainerColor = Color(0xFF0D1117),
                        unfocusedContainerColor = Color(0xFF0D1117)
                    )
                )

                Spacer(modifier = Modifier.width(8.dp))

                IconButton(
                    onClick = {
                        val prompt = inputText.trim()
                        if (prompt.isEmpty() || isStreaming) return@IconButton
                        
                        inputText = ""
                        chatMessages.add(ChatMessage(sender = "user", text = prompt))
                        isStreaming = true
                        activeThinkingText = ""
                        activeImageBitmap = null
                        activeAssistantMessage = ""

                        scope.launch {
                            try {
                                engine.streamRealKimiK3CloudInference(prompt) { tokenText, isThinking, isMedia, bitmap ->
                                    scope.launch(Dispatchers.Main) {
                                        if (isThinking) {
                                            activeThinkingText += tokenText
                                        } else if (isMedia) {
                                            activeImageBitmap = bitmap
                                        } else {
                                            activeAssistantMessage += tokenText
                                        }
                                    }
                                }

                                withContext(Dispatchers.Main) {
                                    chatMessages.add(
                                        ChatMessage(
                                            sender = "assistant",
                                            text = activeAssistantMessage.trim(),
                                            thinkingText = activeThinkingText.trim(),
                                            imageBitmap = activeImageBitmap
                                        )
                                    )
                                    activeThinkingText = ""
                                    activeImageBitmap = null
                                    activeAssistantMessage = ""
                                    isStreaming = false
                                }
                            } catch (e: Throwable) {
                                Log.e("ZeroCopyInfer", "Error during Real Cloud SSE Stream inference", e)
                                withContext(Dispatchers.Main) {
                                    chatMessages.add(ChatMessage(sender = "assistant", text = "[Error]: ${e.localizedMessage}"))
                                    activeThinkingText = ""
                                    activeImageBitmap = null
                                    activeAssistantMessage = ""
                                    isStreaming = false
                                }
                            }
                        }
                    },
                    enabled = !isStreaming && inputText.isNotBlank(),
                    modifier = Modifier
                        .size(48.dp)
                        .background(
                            color = if (!isStreaming && inputText.isNotBlank()) Color(0xFF1F6FEB) else Color(0xFF30363D),
                            shape = RoundedCornerShape(24.dp)
                        )
                ) {
                    if (isStreaming) {
                        CircularProgressIndicator(modifier = Modifier.size(20.dp), color = Color.White, strokeWidth = 2.dp)
                    } else {
                        Icon(imageVector = Icons.AutoMirrored.Filled.Send, contentDescription = "Send", tint = Color.White)
                    }
                }
            }
        }
    }
}

@Composable
fun ChatBubble(msg: ChatMessage) {
    val isUser = msg.sender == "user"
    var isThinkingExpanded by remember { mutableStateOf(true) }

    Box(
        modifier = Modifier.fillMaxWidth(),
        contentAlignment = if (isUser) Alignment.CenterEnd else Alignment.CenterStart
    ) {
        Column(modifier = Modifier.widthIn(max = 300.dp)) {
            // Render Reasoning Accordion Card if thinkingText exists
            if (!isUser && msg.thinkingText.isNotBlank()) {
                Surface(
                    color = Color(0xFF161B22),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 6.dp)
                        .border(1.dp, Color(0xFFD29922).copy(alpha = 0.5f), RoundedCornerShape(12.dp))
                ) {
                    Column(modifier = Modifier.padding(10.dp)) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.clickable { isThinkingExpanded = !isThinkingExpanded }
                        ) {
                            Text(
                                text = "🧠 Pensamiento Kimi-K3 (<|open|>)",
                                color = Color(0xFFD29922),
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold
                            )
                            Spacer(modifier = Modifier.weight(1f))
                            Icon(
                                imageVector = if (isThinkingExpanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                                contentDescription = "Toggle",
                                tint = Color(0xFFD29922)
                            )
                        }

                        AnimatedVisibility(visible = isThinkingExpanded) {
                            Text(
                                text = msg.thinkingText,
                                color = Color(0xFF8B949E),
                                fontSize = 11.sp,
                                fontFamily = FontFamily.Monospace,
                                modifier = Modifier.padding(top = 6.dp)
                            )
                        }
                    }
                }
            }

            // Render Real Generated Bitmap Image Card if imageBitmap exists (<|media_begin|>)
            if (!isUser && msg.imageBitmap != null) {
                Surface(
                    color = Color(0xFF161B22),
                    shape = RoundedCornerShape(14.dp),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 6.dp)
                        .border(1.dp, Color(0xFF238636).copy(alpha = 0.8f), RoundedCornerShape(14.dp))
                ) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Text(
                            text = "🎨 Imagen Generada por Kimi-K3 (<|media_begin|>)",
                            color = Color(0xFF3FB950),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(bottom = 8.dp)
                        )
                        Surface(
                            shape = RoundedCornerShape(8.dp),
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(180.dp)
                        ) {
                            Image(
                                bitmap = msg.imageBitmap.asImageBitmap(),
                                contentDescription = "Generated Image",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier.fillMaxSize()
                            )
                        }
                    }
                }
            }

            // Render Final Text Bubble
            if (msg.text.isNotBlank() || isUser) {
                Surface(
                    color = if (isUser) Color(0xFF1F6FEB) else Color(0xFF21262D),
                    shape = RoundedCornerShape(
                        topStart = 16.dp,
                        topEnd = 16.dp,
                        bottomStart = if (isUser) 16.dp else 4.dp,
                        bottomEnd = if (isUser) 4.dp else 16.dp
                    )
                ) {
                    Text(
                        text = msg.text,
                        color = Color.White,
                        fontSize = 14.sp,
                        modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp)
                    )
                }
            }
        }
    }
}
