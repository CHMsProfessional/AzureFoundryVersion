# Speech example to test the Azure Voice Live API + Gradio UI (fixed)
import os
import uuid
import json
import time
import base64
import logging
import threading
import numpy as np
import sounddevice as sd
import queue
import signal
import sys

from collections import deque
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, ClientSecretCredential, AzureCliCredential, ChainedTokenCredential
import websocket
from websocket import WebSocketApp
from datetime import datetime
import gradio as gr

# =========================
# Global variables
# =========================
stop_event = threading.Event()
logger = logging.getLogger(__name__)

INPUT_SAMPLE_RATE = 48000   # mic -> Voice Live (pcm16_48000hz)
OUTPUT_SAMPLE_RATE = 24000  # playback (pcm16_24000hz)

_connection = None
_send_thread = None
_recv_thread = None

# Cola hacia la UI (para transcripciones y mensajes)
# items: ("user"|"assistant"|"status", text)
ui_events = queue.Queue()

# =========================
# VoiceLiveConnection
# =========================
class VoiceLiveConnection:
    def __init__(self, url: str, headers: list[str]) -> None:
        self._url = url
        self._headers = headers
        self._ws: WebSocketApp | None = None
        self._message_queue = queue.Queue()
        self._connected = False

    def connect(self) -> None:
        logger.info(f"🔗 Intentando conectar WebSocket a: {self._url}")
        logger.debug(f"Headers para conexión: {self._headers}")
        
        def on_message(ws, message):
            logger.debug(f"📨 Mensaje WebSocket recibido (tamaño: {len(message)} bytes)")
            self._message_queue.put(message)

        def on_error(ws, error):
            logger.error(f"❌ WebSocket error: {error}")
            logger.error(f"❌ Tipo de error: {type(error)}")
            logger.error(f"❌ Estado de conexión antes del error: {self._connected}")
            ui_events.put(("status", f"❌ WS error: {error}"))

        def on_close(ws, close_status_code, close_msg):
            logger.warning(f"🔌 WebSocket connection closed - Status code: {close_status_code}, Message: {close_msg}")
            logger.warning(f"🔌 Estado previo de conexión: {self._connected}")
            logger.warning(f"🔌 URL de la conexión: {self._url}")
            self._connected = False
            ui_events.put(("status", f"🔌 Conexión cerrada (código: {close_status_code})"))

        def on_open(ws):
            logger.info(f"✅ WebSocket connection opened successfully")
            logger.info(f"✅ URL conectada: {self._url}")
            self._connected = True
            ui_events.put(("status", "✅ Conectado a Azure Voice Live"))

        self._ws = websocket.WebSocketApp(
            self._url,
            header=self._headers,  # LISTA de strings "Header-Name: value"
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        logger.info(f"🚀 Iniciando WebSocket thread con ping_interval=20")
        self._ws_thread = threading.Thread(target=self._ws.run_forever, kwargs={"ping_interval": 20})
        self._ws_thread.daemon = True
        self._ws_thread.start()
        logger.debug(f"🧵 WebSocket thread iniciado: {self._ws_thread.name}")

        timeout = 15
        start_time = time.time()
        logger.info(f"⏱️ Esperando conexión WebSocket (timeout: {timeout}s)")
        while not self._connected and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        elapsed_time = time.time() - start_time
        if not self._connected:
            logger.error(f"❌ Failed to establish WebSocket connection after {elapsed_time:.2f}s")
            logger.error(f"❌ Thread alive: {self._ws_thread.is_alive()}")
            logger.error(f"❌ WebSocket state: {self._ws}")
            raise ConnectionError(f"Failed to establish WebSocket connection after {elapsed_time:.2f}s")
        else:
            logger.info(f"✅ WebSocket connected successfully in {elapsed_time:.2f}s")

    def recv(self) -> str | None:
        try:
            message = self._message_queue.get(timeout=1)
            logger.debug(f"📨 Mensaje extraído de queue (tamaño: {len(message)} bytes)")
            return message
        except queue.Empty:
            logger.debug("📭 Queue vacía en recv()")
            return None

    def send(self, message: str) -> None:
        if self._ws and self._connected:
            try:
                logger.debug(f"📤 Enviando mensaje WebSocket (tamaño: {len(message)} bytes)")
                self._ws.send(message)
                logger.debug("📤 Mensaje enviado exitosamente")
            except Exception as e:
                logger.error(f"❌ Error enviando mensaje WebSocket: {e}")
                logger.error(f"❌ Estado conexión: {self._connected}")
                logger.error(f"❌ WebSocket válido: {self._ws is not None}")
        else:
            logger.warning(f"⚠️ Intento de envío con WS no conectado - WS válido: {self._ws is not None}, Conectado: {self._connected}")

    def close(self) -> None:
        logger.info("🔚 Cerrando conexión WebSocket...")
        if self._ws:
            try:
                logger.debug("🔚 Cerrando WebSocket...")
                self._ws.close()
                logger.debug("🔚 WebSocket cerrado")
            except Exception as e:
                logger.error(f"❌ Error cerrando WebSocket: {e}")
        self._connected = False
        logger.info("🔚 Conexión marcada como cerrada")

# =========================
# AzureVoiceLive (helper)
# =========================
class AzureVoiceLive:
    def __init__(
        self,
        *,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        token: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._azure_endpoint = (azure_endpoint or "").rstrip("/")
        self._api_version = api_version
        self._token = token
        self._api_key = api_key
        self._connection: VoiceLiveConnection | None = None

    def connect(self, project_name: str, agent_id: str, agent_access_token: str) -> VoiceLiveConnection:
        if self._connection is not None:
            raise ValueError("Already connected to the Voice Live API.")
        if not project_name:
            raise ValueError("Project name is required.")
        if not agent_id:
            raise ValueError("Agent ID is required.")
        if not agent_access_token:
            raise ValueError("Agent access token is required.")

        azure_ws_endpoint = self._azure_endpoint.replace("https://", "wss://")
        url = (
            f"{azure_ws_endpoint}/voice-live/realtime"
            f"?api-version={self._api_version}"
            f"&agent-project-name={project_name}"
            f"&agent-id={agent_id}"
            f"&agent-access-token={agent_access_token}"
            f"&debug=on"
        )

        if self._token:
            auth_header = [f"Authorization: Bearer {self._token}"]
        elif self._api_key:
            auth_header = [f"api-key: {self._api_key}"]
        else:
            raise ValueError("Either token or api_key must be provided.")

        request_id = str(uuid.uuid4())
        headers_list = [f"x-ms-client-request-id: {request_id}", *auth_header]

        self._connection = VoiceLiveConnection(url, headers_list)
        self._connection.connect()
        return self._connection

# =========================
# Audio playback (async queue)
# =========================
class AudioPlayerAsync:
    def __init__(self):
        self.queue = deque()
        self.lock = threading.Lock()
        self.stream = sd.OutputStream(
            callback=self.callback,
            samplerate=OUTPUT_SAMPLE_RATE,
            channels=1,
            dtype=np.int16,
            blocksize=int(OUTPUT_SAMPLE_RATE * 0.1),  # 100 ms
        )
        self.playing = False

    def callback(self, outdata, frames, time_info, status):
        if status:
            logger.warning(f"Stream status: {status}")
        with self.lock:
            data = np.empty(0, dtype=np.int16)
            while len(data) < frames and len(self.queue) > 0:
                item = self.queue.popleft()
                frames_needed = frames - len(data)
                data = np.concatenate((data, item[:frames_needed]))
                if len(item) > frames_needed:
                    self.queue.appendleft(item[frames_needed:])
            if len(data) < frames:
                data = np.concatenate((data, np.zeros(frames - len(data), dtype=np.int16)))
        outdata[:] = data.reshape(-1, 1)

    def add_data(self, data: bytes):
        with self.lock:
            np_data = np.frombuffer(data, dtype=np.int16)
            self.queue.append(np_data)
            if not self.playing and len(self.queue) > 0:
                self.start()

    def start(self):
        if not self.playing:
            self.playing = True
            self.stream.start()

    def stop(self):
        with self.lock:
            self.queue.clear()
        self.playing = False
        self.stream.stop()

    def terminate(self):
        with self.lock:
            self.queue.clear()
        try:
            self.stream.stop()
        finally:
            self.stream.close()

# =========================
# Audio capture & send
# =========================
def listen_and_send_audio(connection: VoiceLiveConnection) -> None:
    logger.info("🎤 Starting audio stream ...")
    logger.debug(f"🎤 Audio config - Sample rate: {INPUT_SAMPLE_RATE}Hz, Read size: {int(INPUT_SAMPLE_RATE * 0.02)} frames")
    ui_events.put(("status", "🎤 Mic activo (48 kHz)"))
    
    stream = sd.InputStream(channels=1, samplerate=INPUT_SAMPLE_RATE, dtype="int16")
    try:
        logger.debug("🎤 Iniciando stream de audio...")
        stream.start()
        logger.info("🎤 Stream de audio iniciado exitosamente")
        
        read_size = int(INPUT_SAMPLE_RATE * 0.02)  # 20ms -> 960 frames @ 48k
        frames_sent = 0
        last_log_time = time.time()
        
        while not stop_event.is_set():
            if stream.read_available >= read_size:
                try:
                    data, _ = stream.read(read_size)  # ndarray int16 shape (N, 1)
                    audio_b64 = base64.b64encode(data.tobytes()).decode("utf-8")
                    param = {"type": "input_audio_buffer.append", "audio": audio_b64, "event_id": ""}
                    
                    connection.send(json.dumps(param))
                    frames_sent += 1
                    
                    # Log cada 5 segundos para no saturar
                    current_time = time.time()
                    if current_time - last_log_time > 5.0:
                        logger.debug(f"🎤 Audio enviado - Frames: {frames_sent}, Duración: {frames_sent * 0.02:.1f}s")
                        last_log_time = current_time
                        
                except Exception as e:
                    logger.error(f"❌ Error procesando/enviando audio frame: {e}")
                    logger.error(f"❌ Frames enviados hasta ahora: {frames_sent}")
            else:
                time.sleep(0.001)
                
    except Exception as e:
        logger.error(f"❌ Audio stream interrupted. {e}")
        logger.error(f"❌ Tipo de error: {type(e)}")
        logger.error(f"❌ Frames enviados antes del error: {frames_sent}")
        ui_events.put(("status", f"❌ Audio stream error: {e}"))
    finally:
        try:
            logger.debug("🛑 Deteniendo stream de audio...")
            stream.stop()
            logger.debug("🛑 Stream de audio detenido")
        finally:
            stream.close()
            logger.info(f"🛑 Audio stream closed. Total frames enviados: {frames_sent}")
        ui_events.put(("status", "🛑 Mic detenido"))

# =========================
# Receive & playback
# =========================
def receive_audio_and_playback(connection: VoiceLiveConnection) -> None:
    last_audio_item_id = None
    audio_player = AudioPlayerAsync()
    ui_events.put(("status", "🔊 Reproducción activada (24 kHz)"))

    logger.info("🔊 Starting audio playback ...")
    logger.debug(f"🔊 Audio playback config - Sample rate: {OUTPUT_SAMPLE_RATE}Hz")
    
    events_received = 0
    audio_chunks_received = 0
    transcriptions_received = 0
    last_log_time = time.time()
    
    try:
        while not stop_event.is_set():
            raw_event = connection.recv()
            if raw_event is None:
                continue

            events_received += 1
            try:
                event = json.loads(raw_event)
                event_type = event.get("type")
                logger.debug(f"📨 Evento recibido: {event_type}")

                if event_type == "session.created":
                    session = event.get("session", {})
                    session_id = session.get('id', 'unknown')
                    logger.info(f"🆔 Session created: {session_id}")
                    logger.debug(f"🆔 Session details: {session}")
                    ui_events.put(("status", f"🆔 Sesión: {session_id}"))

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    text = event.get("transcript", "") or ""
                    transcriptions_received += 1
                    if text.strip():
                        logger.info(f"👤 User transcription: {text}")
                        ui_events.put(("user", text))

                elif event_type == "response.text.done":
                    text = event.get("text", "") or ""
                    if text.strip():
                        logger.info(f"🤖 Assistant text: {text}")
                        ui_events.put(("assistant", text))

                elif event_type == "response.audio_transcript.done":
                    text = event.get("transcript", "") or ""
                    if text.strip():
                        logger.info(f"🤖 Assistant audio transcript: {text}")
                        ui_events.put(("assistant", text))

                elif event_type == "response.audio.delta":
                    item_id = event.get("item_id")
                    if item_id != last_audio_item_id:
                        last_audio_item_id = item_id
                        logger.debug(f"🔊 Nuevo audio item: {item_id}")
                    
                    b64 = event.get("delta") or event.get("audio") or ""
                    if b64:
                        audio_chunks_received += 1
                        bytes_data = base64.b64decode(b64)
                        audio_player.add_data(bytes_data)
                        logger.debug(f"🔊 Audio chunk procesado: {len(bytes_data)} bytes")

                elif event_type == "input_audio_buffer.speech_started":
                    logger.info("👂 Speech detection: Usuario empezó a hablar")
                    ui_events.put(("status", "👂 Detectado habla del usuario"))
                    audio_player.stop()

                elif event_type == "error":
                    err = event.get("error", {})
                    error_type = err.get('type', 'Unknown')
                    error_code = err.get('code', 'Unknown')
                    error_message = err.get('message', '')
                    msg = f"Error received: Type={error_type} Code={error_code} Msg={error_message}"
                    logger.error(f"❌ {msg}")
                    logger.error(f"❌ Full error event: {event}")
                    ui_events.put(("status", f"❌ {msg}"))

                # Log estadísticas cada 10 segundos
                current_time = time.time()
                if current_time - last_log_time > 10.0:
                    logger.info(f"📊 Stats - Events: {events_received}, Audio chunks: {audio_chunks_received}, Transcriptions: {transcriptions_received}")
                    last_log_time = current_time

            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse JSON event: {e}")
                logger.error(f"❌ Raw event (primeros 200 chars): {raw_event[:200]}")
                continue

    except Exception as e:
        logger.error(f"❌ Error in audio playback: {e}")
        logger.error(f"❌ Tipo de error: {type(e)}")
        logger.error(f"❌ Events procesados antes del error: {events_received}")
        ui_events.put(("status", f"❌ Playback error: {e}"))
    finally:
        logger.info(f"🔇 Terminando playback - Events: {events_received}, Audio chunks: {audio_chunks_received}")
        audio_player.terminate()
        logger.info("🔇 Playback done.")
        ui_events.put(("status", "🔇 Reproducción detenida"))

# =========================
# Session control
# =========================
def start_session():
    """
    Inicializa credenciales, conecta WS, envía session.update y levanta hilos de audio.
    """
    global _connection, _send_thread, _recv_thread
    logger.info("🚀 === INICIANDO SESIÓN DE VOICE LIVE ===")
    
    try:
        stop_event.clear()
        logger.debug("🚀 Stop event cleared")

        logger.info("📁 Cargando variables de entorno...")
        load_dotenv("./.env", override=True)
        endpoint = os.environ.get("AZURE_VOICE_LIVE_ENDPOINT") or "https://your-endpoint.azure.com"
        agent_id = os.environ.get("AI_FOUNDRY_AGENT_ID") or "your-agent-id"
        project_name = os.environ.get("AI_FOUNDRY_PROJECT_NAME") or "your-project-name"
        api_version = os.environ.get("AZURE_VOICE_LIVE_API_VERSION") or "2025-05-01-preview"
        api_key = os.environ.get("AZURE_VOICE_LIVE_API_KEY") or None

        logger.info(f"🔧 Config - Endpoint: {endpoint}")
        logger.info(f"🔧 Config - Agent ID: {agent_id}")
        logger.info(f"🔧 Config - Project: {project_name}")
        logger.info(f"🔧 Config - API Version: {api_version}")
        logger.debug(f"🔧 Config - API Key disponible: {api_key is not None}")

        logger.info("🔑 Obteniendo token de Azure...")
        
        # Intentar diferentes métodos de autenticación
        token = None
        auth_method = None
        
        try:
            # Método 1: Intentar con Azure CLI primero (más simple)
            logger.debug("🔑 Intentando autenticación con Azure CLI...")
            cli_cred = AzureCliCredential()
            token = cli_cred.get_token("https://ai.azure.com/.default")
            auth_method = "Azure CLI"
            logger.info("✅ Autenticación exitosa con Azure CLI")
            
        except Exception as cli_error:
            logger.warning(f"⚠️ Azure CLI auth falló: {cli_error}")
            
            try:
                # Método 2: Usar DefaultAzureCredential con configuración específica
                logger.debug("🔑 Intentando con DefaultAzureCredential...")
                # Excluir métodos que requieren PowerShell
                cred = DefaultAzureCredential(
                    exclude_powershell_credential=True,
                    exclude_visual_studio_code_credential=True,
                    exclude_interactive_browser_credential=True
                )
                token = cred.get_token("https://ai.azure.com/.default")
                auth_method = "Default Azure Credential"
                logger.info("✅ Autenticación exitosa con DefaultAzureCredential")
                
            except Exception as default_error:
                logger.error(f"❌ DefaultAzureCredential falló: {default_error}")
                
                # Método 3: Si hay credenciales de Service Principal
                tenant_id = os.environ.get("AZURE_TENANT_ID")
                client_id = os.environ.get("AZURE_CLIENT_ID")
                client_secret = os.environ.get("AZURE_CLIENT_SECRET")
                
                if tenant_id and client_id and client_secret:
                    try:
                        logger.debug("🔑 Intentando con Service Principal...")
                        sp_cred = ClientSecretCredential(tenant_id, client_id, client_secret)
                        token = sp_cred.get_token("https://ai.azure.com/.default")
                        auth_method = "Service Principal"
                        logger.info("✅ Autenticación exitosa con Service Principal")
                    except Exception as sp_error:
                        logger.error(f"❌ Service Principal auth falló: {sp_error}")
                
                if token is None:
                    error_msg = (
                        "❌ No se pudo autenticar con Azure. Opciones:\n"
                        "1. Ejecutar 'az login' en la terminal\n"
                        "2. Configurar variables de entorno: AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET\n"
                        "3. Instalar Az PowerShell: Install-Module -Name Az -Force"
                    )
                    logger.error(error_msg)
                    raise Exception(error_msg)
        
        if token:
            logger.info(f"✅ Token de Azure obtenido exitosamente usando: {auth_method}")
            logger.debug(f"🔑 Token expira en: {token.expires_on}")
        else:
            raise Exception("No se pudo obtener token de Azure")

        logger.info("🏗️ Creando cliente AzureVoiceLive...")
        client = AzureVoiceLive(
            azure_endpoint=endpoint,
            api_version=api_version,
            token=token.token,
            # api_key=api_key,
        )

        logger.info("🔗 Conectando a Azure Voice Live...")
        _connection = client.connect(
            project_name=project_name,
            agent_id=agent_id,
            agent_access_token=token.token
        )
        logger.info("✅ Conexión WebSocket establecida")

        logger.info("⚙️ Enviando configuración de sesión...")
        session_update = {
            "type": "session.update",
            "session": {
                "turn_detection": {
                    "type": "azure_semantic_vad",
                    "threshold": 0.3,
                    "prefix_padding_ms": 200,
                    "silence_duration_ms": 200,
                    "remove_filler_words": False,
                    "end_of_utterance_detection": {"model": "semantic_detection_v1", "threshold": 0.01, "timeout": 2},
                },
                "input_audio_noise_reduction": {"type": "azure_deep_noise_suppression"},
                "input_audio_echo_cancellation": {"type": "server_echo_cancellation"},
                "voice": {"name": "en-US-Ava:DragonHDLatestNeural", "type": "azure-standard", "temperature": 0.8},
                "input_audio_format":  "pcm16_48000hz",
                "output_audio_format": "pcm16_24000hz",
            },
            "event_id": ""
        }
        logger.debug(f"⚙️ Session config: {json.dumps(session_update, indent=2)}")
        _connection.send(json.dumps(session_update))
        logger.info("✅ Configuración de sesión enviada")
        ui_events.put(("status", "⚙️ Sesión configurada"))

        logger.info("🧵 Iniciando hilos de audio...")
        _send_thread = threading.Thread(target=listen_and_send_audio, args=(_connection,), daemon=True, name="AudioSend")
        _recv_thread = threading.Thread(target=receive_audio_and_playback, args=(_connection,), daemon=True, name="AudioRecv")
        
        _send_thread.start()
        _recv_thread.start()
        
        logger.info(f"🧵 Hilos iniciados - Send: {_send_thread.name}, Recv: {_recv_thread.name}")
        logger.info("✅ === SESIÓN INICIADA EXITOSAMENTE ===")
        
    except Exception as e:
        logger.error(f"❌ === ERROR AL INICIAR SESIÓN ===")
        logger.error(f"❌ Error: {e}")
        logger.error(f"❌ Tipo: {type(e)}")
        logger.exception("❌ Stack trace completo:")
        ui_events.put(("status", f"❌ Error iniciando sesión: {e}"))
        raise

def stop_session():
    """
    Señaliza stop y cierra WS.
    """
    global _connection, _send_thread, _recv_thread
    logger.info("🛑 === DETENIENDO SESIÓN DE VOICE LIVE ===")
    
    try:
        logger.debug("🛑 Configurando stop event...")
        stop_event.set()
        
        logger.debug("🛑 Esperando 0.2s para que los hilos se detengan...")
        time.sleep(0.2)
        
        # Verificar estado de hilos
        if _send_thread:
            logger.debug(f"🧵 Send thread alive: {_send_thread.is_alive()}")
        if _recv_thread:
            logger.debug(f"🧵 Recv thread alive: {_recv_thread.is_alive()}")
        
        logger.info("🔌 Cerrando conexión WebSocket...")
        if _connection:
            _connection.close()
            logger.info("✅ Conexión WebSocket cerrada")
        else:
            logger.warning("⚠️ No había conexión activa para cerrar")
            
    except Exception as e:
        logger.error(f"❌ Error durante stop_session: {e}")
        logger.error(f"❌ Tipo: {type(e)}")
        logger.exception("❌ Stack trace:")
    finally:
        _connection = None
        logger.info("🧹 Variables globales limpiadas")
        logger.info("✅ === SESIÓN DETENIDA ===")
    
    ui_events.put(("status", "🧹 Sesión finalizada"))

# =========================
# Gradio UI helpers
# =========================
def gr_start(chat_messages):
    # chat_messages es una lista de dicts: {role, content}
    if _connection is not None:
        chat_messages.append({"role": "assistant", "content": "Ya está conectado."})
        return chat_messages
    try:
        start_session()
        # Añadimos un status como assistant
        chat_messages.append({"role": "assistant", "content": "Iniciando voz…"})
        return chat_messages
    except Exception as e:
        chat_messages.append({"role": "assistant", "content": f"Error al iniciar: {e}"})
        return chat_messages

def gr_stop(chat_messages):
    if _connection is None:
        chat_messages.append({"role": "assistant", "content": "No hay sesión activa."})
        return chat_messages
    try:
        stop_session()
        chat_messages.append({"role": "assistant", "content": "Sesión detenida."})
        return chat_messages
    except Exception as e:
        chat_messages.append({"role": "assistant", "content": f"Error al detener: {e}"})
        return chat_messages

def gr_poll(chat_messages):
    """
    Drena eventos de ui_events y actualiza el Chatbot.
    Chatbot (type='messages') espera: { "role": "user"|"assistant", "content": str }
    """
    updated = False
    while True:
        try:
            kind, text = ui_events.get_nowait()
        except queue.Empty:
            break
        updated = True
        if kind == "user":
            chat_messages.append({"role": "user", "content": text})
        elif kind == "assistant":
            chat_messages.append({"role": "assistant", "content": text})
        elif kind == "status":
            chat_messages.append({"role": "assistant", "content": f"[Estado] {text}"})
    return chat_messages if updated else gr.update()

# =========================
# Log helper
# =========================
def write_conversation_log(message: str) -> None:
    with open(f'logs/{logfilename}', 'a', encoding="utf-8") as conversation_log:
        conversation_log.write(message + "\n")

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    try:
        # Directorio del script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        print(f"Working directory: {script_dir}")

        # Carpeta logs
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # Archivos de log con timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logfilename = f"{timestamp}_conversation.log"
        log_filepath = f'logs/{timestamp}_voicelive.log'

        logging.basicConfig(
            filename=log_filepath,
            filemode="w",
            level=logging.DEBUG,
            format='%(asctime)s:%(name)s:%(levelname)s:%(message)s'
        )
        
        # También log a consola para debug inmediato
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        logger.info(f"🔧 === INICIANDO AZURE VOICE LIVE CHATBOT ===")
        logger.info(f"🔧 Script directory: {script_dir}")
        logger.info(f"🔧 Log file: {log_filepath}")
        logger.info(f"🔧 Python version: {sys.version}")
        logger.info(f"🔧 Gradio version: {gr.__version__}")
        
        # Log información del entorno
        logger.debug(f"🔧 Environment variables relevant:")
        env_vars = [
            "AZURE_VOICE_LIVE_ENDPOINT", "AI_FOUNDRY_AGENT_ID", "AI_FOUNDRY_PROJECT_NAME", 
            "AZURE_VOICE_LIVE_API_VERSION", "AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"
        ]
        for key in env_vars:
            value = os.environ.get(key, "NOT SET")
            # No logear valores sensibles completos
            if value != "NOT SET" and len(value) > 10:
                logged_value = f"{value[:10]}...{value[-4:]}"
            else:
                logged_value = value
            logger.debug(f"🔧   {key}: {logged_value}")

        # Señales para cierre limpio
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received interrupt signal {signum}, shutting down...")
            print(f"\nReceived interrupt signal {signum}, shutting down...")
            stop_event.set()
            try:
                stop_session()
            except Exception as e:
                logger.error(f"❌ Error during signal cleanup: {e}")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info("🚀 Iniciando interfaz Gradio...")

        # ========= Gradio UI =========
        with gr.Blocks(title="Azure Voice Live - Voice Chat") as demo:
            gr.Markdown("## 🎙️ Azure Voice Live (Agente)\nPulsa **Iniciar** para empezar a hablar. Pulsa **Detener** para terminar.")

            with gr.Row():
                start_btn = gr.Button("▶️ Iniciar")
                stop_btn = gr.Button("⏹️ Detener")

            # type='messages' => lista de dicts {role, content}
            chat = gr.Chatbot(label="Transcripciones (usuario ↔ agente)", height=420, type='messages')
            state = gr.State([])  # almacenará la lista de mensajes

            start_btn.click(gr_start, inputs=[state], outputs=[state])
            stop_btn.click(gr_stop, inputs=[state], outputs=[state])

            # Timer para refrescar la conversación
            demo.load(lambda: [], None, state)  # init vacío
            timer = gr.Timer(0.5)
            timer.tick(gr_poll, inputs=[state], outputs=[state], show_progress=False)
            # Refleja state -> chat en cada tick/load
            def mirror(messages): return messages
            demo.load(mirror, inputs=[state], outputs=[chat])
            timer.tick(mirror, inputs=[state], outputs=[chat], show_progress=False)

        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
        logger.info("🏁 Gradio app launched successfully")

    except Exception as e:
        logger.error(f"❌ === ERROR CRÍTICO EN MAIN ===")
        logger.error(f"❌ Error: {e}")
        logger.error(f"❌ Tipo: {type(e)}")
        logger.exception("❌ Stack trace completo:")
        print(f"Error: {e}")
        stop_event.set()
