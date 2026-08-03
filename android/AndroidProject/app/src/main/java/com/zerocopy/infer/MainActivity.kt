package com.zerocopy.infer

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
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
    var activeAssistantMessage by remember { mutableStateOf("") }
    
    val listState = rememberLazyListState()
    val scope = rememberCoroutineScope()

    LaunchedEffect(chatMessages.size, activeAssistantMessage) {
        if (chatMessages.isNotEmpty() || activeAssistantMessage.isNotEmpty()) {
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
                        text = "ZeroCopy-Infer Chat",
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
                            text = "0 Bytes SSD",
                            color = Color.White,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
                        )
                    }
                }
                Text(
                    text = "Kimi K3 • Continuous Cloud HTTP Range Stream • Leandro Timberini",
                    fontSize = 11.sp,
                    color = Color.Gray,
                    modifier = Modifier.padding(top = 2.dp)
                )
            }
        }

        // Chat Message Area
        LazyColumn(
            state = listState,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
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
                            text = "¡Hola! Escribe cualquier pregunta para comenzar una conversación fluida con ZeroCopy-Infer.",
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
                    ChatBubble(ChatMessage(sender = "assistant", text = activeAssistantMessage))
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
                    placeholder = { Text("Escribe tu mensaje...", color = Color.Gray) },
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
                        activeAssistantMessage = ""

                        scope.launch {
                            try {
                                withContext(Dispatchers.IO) {
                                    val promptIds = prompt.split(" ").map { it.hashCode() and 0xFFFF }.toIntArray()
                                    val numTokens = 15
                                    var currentText = ""

                                    for (i in 1..numTokens) {
                                        val res = engine.streamTokenDynamic(prompt, promptIds, i)
                                        currentText += "${res.decodedWord} "

                                        withContext(Dispatchers.Main) {
                                            activeAssistantMessage = currentText
                                        }
                                    }

                                    withContext(Dispatchers.Main) {
                                        chatMessages.add(ChatMessage(sender = "assistant", text = currentText.trim()))
                                        activeAssistantMessage = ""
                                        isStreaming = false
                                    }
                                }
                            } catch (e: Throwable) {
                                Log.e("ZeroCopyInfer", "Error in chat streaming", e)
                                withContext(Dispatchers.Main) {
                                    chatMessages.add(ChatMessage(sender = "assistant", text = "[Error]: ${e.localizedMessage}"))
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
                        Icon(imageVector = Icons.Default.Send, contentDescription = "Send", tint = Color.White)
                    }
                }
            }
        }
    }
}

@Composable
fun ChatBubble(msg: ChatMessage) {
    val isUser = msg.sender == "user"
    Box(
        modifier = Modifier.fillMaxWidth(),
        contentAlignment = if (isUser) Alignment.CenterEnd else Alignment.CenterStart
    ) {
        Surface(
            color = if (isUser) Color(0xFF1F6FEB) else Color(0xFF21262D),
            shape = RoundedCornerShape(
                topStart = 16.dp,
                topEnd = 16.dp,
                bottomStart = if (isUser) 16.dp else 4.dp,
                bottomEnd = if (isUser) 4.dp else 16.dp
            ),
            modifier = Modifier.widthIn(max = 280.dp)
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
