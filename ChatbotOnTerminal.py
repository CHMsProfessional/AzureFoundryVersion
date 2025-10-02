# ==========================================================
# IMPORTANT: Before running this program, make sure to log in
# to your Azure account using the command:
#     az login
# Otherwise, the program will not be able to obtain permissions.
# ==========================================================

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
import re

from collections import deque
from dotenv import load_dotenv
from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from typing import Dict, Union, Literal, Set
from typing_extensions import Iterator, TypedDict, Required
import websocket
from websocket import WebSocketApp
from datetime import datetime

# =========================
# Globals / coordination
# =========================
stop_event = threading.Event()
connection_queue = queue.Queue()

logger = logging.getLogger(__name__)
AUDIO_SAMPLE_RATE = 24000

# --- Voice catalog (optional) ---
AZURE_VOICES = {
    # === English (US) Dragon HD ===
    "en-US-Ava:DragonHDLatestNeural": {"locale": "en-US", "gender": "Female"},
    "en-US-Andrew:DragonHDLatestNeural": {"locale": "en-US", "gender": "Male"},
    "en-US-Alloy:DragonHDLatestNeural": {"locale": "en-US", "gender": "Male"},   #  
    
    # === English (UK) Neural ===
    "en-GB-OliverNeural": {"locale": "en-GB", "gender": "Male"},   #  
    "en-GB-OliviaNeural": {"locale": "en-GB", "gender": "Female"}, #  

    # === Spanish (Spain) Neural ===
    "es-ES-ElviraNeural": {"locale": "es-ES", "gender": "Female"}, #  
    "es-ES-IreneNeural": {"locale": "es-ES", "gender": "Female"},  #  
    "es-ES-Tristan:DragonHDLatestNeural": {"locale": "es-ES", "gender": "Male"}, #  

    # === Spanish (Mexico) Neural ===
    "es-MX-DaliaNeural": {"locale": "es-MX", "gender": "Female"},  #  
    "es-MX-JorgeNeural": {"locale": "es-MX", "gender": "Male"},    #  
    "es-MX-NuriaNeural": {"locale": "es-MX", "gender": "Female"},  #  

    # === Spanish (LatAm / Other) Neural (optionally highlighted) ===
    "es-CO-SalomeNeural": {"locale": "es-CO", "gender": "Female"}, #  
    "es-CO-GonzaloNeural": {"locale": "es-CO", "gender": "Male"},  #  
    "es-US-PalomaNeural": {"locale": "es-US", "gender": "Female"}, #  
    "es-US-AlonsoNeural": {"locale": "es-US", "gender": "Male"},   #  
}


# Runtime state/context
RESPONSE_ACTIVE = False
RESTART_LOCK = threading.Lock()
PENDING_SHUTDOWN = False
FAREWELL_TIMEOUT_SEC = 8.0
LAST_FAREWELL_TEXT = "It was a pleasure talking to you. See you later! 👋"

RUNTIME = {
    "endpoint": None,
    "project_name": None,
    "agent_id": None,
    "api_version": None,
    "token": None,  # string
}

current_session_cfg = {
    "voice": {
        "name": "en-US-Ava:DragonHDLatestNeural",
        "type": "azure-standard",
        "temperature": 0.8,
    },
    "turn_detection": {
        "type": "azure_semantic_vad",
        "threshold": 0.30,
        "prefix_padding_ms": 200,
        "silence_duration_ms": 200,
        "remove_filler_words": False,
        "end_of_utterance_detection": {
            "model": "semantic_detection_v1",
            "threshold": 0.01,
            "timeout": 2.0,
        },
    },
    "input_audio_noise_reduction": {"type": "azure_deep_noise_suppression"},
    "input_audio_echo_cancellation": {"type": "server_echo_cancellation"},
}

# =========================
# Topics / local shortcuts
# =========================
# mode: "local" => does NOT use the agent (no TTS), executes local action
#       "agent" => sends a short instruction to the agent (low tokenization)
#       "agent_then_shutdown" => farewell TTS and shutdown
TOPICS = [
    # {
    #     "name": "farewell_shutdown",
    #     "pattern": r"\b(adios|adiós|hasta luego|chao|chau|nos vemos)\b",
    #     "mode": "agent_then_shutdown",
    #     "reply": "It was a pleasure talking to you. See you later! 👋",
    #     "action": "shutdown",
    # },
    # {
    #     "name": "help_basic",
    #     "pattern": r"\b(ayuda|help)\b",
    #     "mode": "agent",
    #     "reply": "You can ask me to set reminders, answer questions, or process audio. How can I help you?",
    #     "action": None,
    # },
]

def match_topic(text: str):
    """
    Checks if the input text matches any predefined topic pattern.
    Returns the matched topic dict or None.
    """
    if not text:
        return None
    low = text.lower()
    for t in TOPICS:
        if re.search(t["pattern"], low):
            return t
    return None

def handle_topic_local(topic: dict):
    """
    Handles a local topic by printing its reply and performing its action.
    Used for local-only commands (no agent interaction).
    """
    reply = topic.get("reply", "")
    if reply:
        print(f"\n\t[TOPIC:LOCAL] {reply}\n")
        write_conversation_log(f"Topic Local Reply: {reply}")
    action = topic.get("action")
    if action == "shutdown":
        stop_event.set()

def handle_topic_agent(connection, topic: dict):
    """
    Sends a reply to the agent for a matched topic using the WebSocket connection.
    Used for agent-based commands.
    """
    reply = topic.get("reply", "")
    if not reply:
        return
    payload = {
        "type": "response.create",
        "response": {
            "instructions": f"Say it exactly and naturally: {reply}"
        },
        "event_id": ""
    }
    try:
        connection.send(json.dumps(payload))
        write_conversation_log(f"Topic Agent Reply sent: {json.dumps(payload)}")
    except Exception as e:
        logger.error(f"Could not send topic agent reply: {e}")

# =========================
# Farewell helpers: cancel, wait, retry
# =========================
def _wait_until_no_active_response(timeout_s: float = 2.5, poll_ms: int = 50) -> bool:
    """
    Waits until there is no active response or until timeout.
    Returns True if inactive, False otherwise.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not globals().get("RESPONSE_ACTIVE", False):
            return True
        time.sleep(poll_ms / 1000.0)
    return not globals().get("RESPONSE_ACTIVE", False)

def _retry_farewell_when_idle(connection, text: str, max_wait_s: float = 3.0):
    """
    Retries sending a farewell message when there is no active response.
    Used to ensure the farewell is delivered.
    """
    def _attempt():
        if _wait_until_no_active_response(timeout_s=max_wait_s):
            try:
                payload = {
                    "type": "response.create",
                    "response": {"instructions": f"Say it exactly and with a cordial tone: {text}"},
                    "event_id": ""
                }
                connection.send(json.dumps(payload))
                write_conversation_log(f"[Farewell RETRY] sent: {json.dumps(payload)}")
            except Exception as e:
                logger.error(f"Retry farewell failed: {e}")
        else:
            logger.warning("Could not retry farewell: active response present.")
    t = threading.Timer(0.2, _attempt)  # small delay to process cancel
    t.daemon = True
    t.start()

def say_then_shutdown(connection, text: str, timeout_s: float = None):
    """
    Cancels any active response, waits for inactivity, sends a farewell TTS,
    and then shuts down after playback or timeout.
    """
    global PENDING_SHUTDOWN, LAST_FAREWELL_TEXT
    if not text:
        text = "See you later. Shutting down now."
    LAST_FAREWELL_TEXT = text

    # 1) Try to cancel any ongoing audio to prioritize farewell
    try:
        connection.send(json.dumps({"type": "response.cancel", "event_id": ""}))
        time.sleep(0.12)  # brief pause for backend to process cancel
    except Exception:
        pass

    # 2) Wait until there is really no active response
    _wait_until_no_active_response(timeout_s=2.5, poll_ms=40)

    # 3) Send the farewell (if fails due to active response, will retry)
    payload = {
        "type": "response.create",
        "response": {"instructions": f"Say it exactly and in a cordial tone: {text}"},
        "event_id": ""
    }
    try:
        connection.send(json.dumps(payload))
        write_conversation_log(f"[Farewell] sent: {json.dumps(payload)}")
        globals()["PENDING_SHUTDOWN"] = True
    except Exception as e:
        logger.error(f"Could not send farewell TTS (first attempt): {e}")
        globals()["PENDING_SHUTDOWN"] = True
        _retry_farewell_when_idle(connection, text, max_wait_s=3.0)

    # 4) Timeout fallback
    t = timeout_s if timeout_s is not None else FAREWELL_TIMEOUT_SEC
    if t and t > 0:
        def _force_shutdown():
            if globals().get("PENDING_SHUTDOWN", False):
                logger.info("Farewell timeout: force shutdown.")
                stop_event.set()
        timer = threading.Timer(t, _force_shutdown)
        timer.daemon = True
        timer.start()

# =========================
# Configuration helpers
# =========================
def _env_bool(key: str, default: bool) -> bool:
    """
    Reads a boolean value from environment variables.
    Returns the value or the default if not set.
    """
    val = os.getenv(key)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")

def _env_float(key: str, default: float) -> float:
    """
    Reads a float value from environment variables.
    Returns the value or the default if not set.
    """
    try:
        return float(os.getenv(key, default))
    except Exception:
        return default

def _env_int(key: str, default: int) -> int:
    """
    Reads an integer value from environment variables.
    Returns the value or the default if not set.
    """
    try:
        return int(os.getenv(key, default))
    except Exception:
        return default

def _env_str(key: str, default: str) -> str:
    """
    Reads a string value from environment variables.
    Returns the value or the default if not set.
    """
    val = os.getenv(key)
    return default if val is None else val.strip()

def _apply_preset(config: dict, name: str) -> dict:
    """
    Applies a preset configuration to the session config dictionary.
    Returns the updated config.
    """
    presets = {
        "callcenter": {
            "turn_detection.threshold": 0.25,
            "turn_detection.silence_ms": 180,
            "voice.temperature": 0.6,
        },
        "ambiente-ruidoso": {
            "turn_detection.threshold": 0.35,
            "turn_detection.silence_ms": 280,
            "voice.temperature": 0.7,
        },
        "dictado": {
            "turn_detection.threshold": 0.20,
            "turn_detection.silence_ms": 350,
            "voice.temperature": 0.3,
        },
    }
    p = presets.get(name)
    if not p:
        return config
    config.setdefault("turn_detection.threshold", p["turn_detection.threshold"])
    config.setdefault("turn_detection.silence_ms", p["turn_detection.silence_ms"])
    config.setdefault("voice.temperature", p["voice.temperature"])
    return config

def build_session_update_from_env() -> dict:
    """
    Builds the session configuration from environment variables and presets.
    Returns the session update payload.
    """
    global current_session_cfg

    cfg = {
        "voice.name": _env_str("VOICE_NAME", current_session_cfg["voice"]["name"]),
        "voice.type": _env_str("VOICE_TYPE", current_session_cfg["voice"]["type"]),
        "voice.temperature": _env_float("VOICE_TEMPERATURE", current_session_cfg["voice"]["temperature"]),
        "turn_detection.threshold": _env_float("VAD_THRESHOLD", current_session_cfg["turn_detection"]["threshold"]),
        "turn_detection.prefix_ms": _env_int("VAD_PREFIX_PADDING_MS", current_session_cfg["turn_detection"]["prefix_padding_ms"]),
        "turn_detection.silence_ms": _env_int("VAD_SILENCE_MS", current_session_cfg["turn_detection"]["silence_duration_ms"]),
        "turn_detection.remove_fillers": _env_bool("VAD_REMOVE_FILLERS", current_session_cfg["turn_detection"]["remove_filler_words"]),
        "eou.model": _env_str("EoU_MODEL", current_session_cfg["turn_detection"]["end_of_utterance_detection"]["model"]),
        "eou.threshold": _env_float("EoU_THRESHOLD", current_session_cfg["turn_detection"]["end_of_utterance_detection"]["threshold"]),
        "eou.timeout_s": _env_float("EoU_TIMEOUT_S", current_session_cfg["turn_detection"]["end_of_utterance_detection"]["timeout"]),
        "nr.type": _env_str("NR_TYPE", current_session_cfg.get("input_audio_noise_reduction", {}).get("type", "")),
        "aec.type": _env_str("AEC_TYPE", current_session_cfg.get("input_audio_echo_cancellation", {}).get("type", "")),
    }

    preset = os.getenv("PRESET")
    if preset:
        cfg = _apply_preset(cfg, preset)

    new_session = {
        "voice": {
            "name": cfg["voice.name"],
            "type": cfg["voice.type"],
            "temperature": float(cfg["voice.temperature"]),
        },
        "turn_detection": {
            "type": "azure_semantic_vad",
            "threshold": float(cfg["turn_detection.threshold"]),
            "prefix_padding_ms": int(cfg["turn_detection.prefix_ms"]),
            "silence_duration_ms": int(cfg["turn_detection.silence_ms"]),
            "remove_filler_words": bool(cfg["turn_detection.remove_fillers"]),
            "end_of_utterance_detection": {
                "model": cfg["eou.model"],
                "threshold": float(cfg["eou.threshold"]),
                "timeout": float(cfg["eou.timeout_s"]),
            },
        },
    }

    if cfg["nr.type"]:
        new_session["input_audio_noise_reduction"] = {"type": cfg["nr.type"]}
    if cfg["aec.type"]:
        new_session["input_audio_echo_cancellation"] = {"type": cfg["aec.type"]}

    current_session_cfg = new_session
    return {
        "type": "session.update",
        "session": new_session,
        "event_id": "",
    }

# =========================
# Client and WS connection
# =========================
class VoiceLiveConnection:
    """
    Manages the WebSocket connection to Azure Voice Live.
    Handles sending, receiving, and connection lifecycle.
    """
    def __init__(self, url: str, headers: dict) -> None:
        self._url = url
        self._headers = headers
        self._ws = None
        self._message_queue = queue.Queue()
        self._connected = False

    def connect(self) -> None:
        def on_message(ws, message):
            self._message_queue.put(message)

        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            logger.info("WebSocket connection closed")
            self._connected = False

        def on_open(ws):
            logger.info("WebSocket connection opened")
            self._connected = True

        self._ws = websocket.WebSocketApp(
            self._url,
            header=self._headers,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )

        self._ws_thread = threading.Thread(target=self._ws.run_forever)
        self._ws_thread.daemon = True
        self._ws_thread.start()

        timeout = 10
        start_time = time.time()
        while not self._connected and time.time() - start_time < timeout:
            time.sleep(0.1)

        if not self._connected:
            raise ConnectionError("Failed to establish WebSocket connection")

    def recv(self) -> str:
        try:
            return self._message_queue.get(timeout=1)
        except queue.Empty:
            return None

    def send(self, message: str) -> None:
        if self._ws and self._connected:
            self._ws.send(message)

    def close(self) -> None:
        if self._ws:
            self._ws.close()
            self._connected = False

class AzureVoiceLive:
    """
    Client for Azure Voice Live API.
    Handles connection setup and authentication.
    """
    def __init__(
        self,
        *,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        token: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._azure_endpoint = azure_endpoint
        self._api_version = api_version
        self._token = token
        self._api_key = api_key
        self._connection = None

    def connect(self, project_name: str, agent_id: str, agent_access_token: str) -> VoiceLiveConnection:
        if self._connection is not None:
            raise ValueError("Already connected to the Voice Live API.")
        if not project_name:
            raise ValueError("Project name is required.")
        if not agent_id:
            raise ValueError("Agent ID is required.")
        if not agent_access_token:
            raise ValueError("Agent access token is required.")

        azure_ws_endpoint = self._azure_endpoint.rstrip('/').replace("https://", "wss://")
        url = f"{azure_ws_endpoint}/voice-live/realtime?api-version={self._api_version}&agent-project-name={project_name}&agent-id={agent_id}&agent-access-token={agent_access_token}"

        auth_header = {"Authorization": f"Bearer {self._token}"} if self._token else {"api-key": self._api_key}
        request_id = uuid.uuid4()
        headers = {"x-ms-client-request-id": str(request_id), **auth_header}

        self._connection = VoiceLiveConnection(url, headers)
        self._connection.connect()
        return self._connection

# =========================
# Audio (playback / capture)
# =========================
class AudioPlayerAsync:
    """
    Asynchronous audio playback handler using sounddevice.
    Buffers and plays audio received from the agent.
    """
    def __init__(self):
        self.queue = deque()
        self.lock = threading.Lock()
        self.stream = sd.OutputStream(
            callback=self.callback,
            samplerate=AUDIO_SAMPLE_RATE,
            channels=1,
            dtype=np.int16,
            blocksize=2400,
        )
        self.playing = False

    def callback(self, outdata, frames, time, status):
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
        self.stream.stop()
        self.stream.close()

def listen_and_send_audio(initial_connection: "VoiceLiveConnection") -> None:
    """
    Captures microphone audio and sends it to the agent in real-time.
    Runs in a separate thread.
    """
    logger.info("Starting audio stream ...")
    stream = sd.InputStream(channels=1, samplerate=AUDIO_SAMPLE_RATE, dtype="int16")
    try:
        stream.start()
        read_size = int(AUDIO_SAMPLE_RATE * 0.02)
        while not stop_event.is_set():
            # Adopt current connection in case of restart
            connection = globals().get("connection", initial_connection)
            if stream.read_available >= read_size:
                data, _ = stream.read(read_size)
                audio = base64.b64encode(data).decode("utf-8")
                param = {"type": "input_audio_buffer.append", "audio": audio, "event_id": ""}
                try:
                    connection.send(json.dumps(param))
                except Exception as e:
                    logger.warning(f"Could not send audio to WS (restart in progress?): {e}")
            else:
                time.sleep(0.001)
    except Exception as e:
        logger.error(f"Audio stream interrupted. {e}")
    finally:
        stream.stop()
        stream.close()
        logger.info("Audio stream closed.")

def receive_audio_and_playback(initial_connection: "VoiceLiveConnection") -> None:
    """
    Receives events and audio from the agent, plays audio, and handles responses.
    Runs in a separate thread.
    """
    last_audio_item_id = None
    audio_player = AudioPlayerAsync()
    global RESPONSE_ACTIVE

    logger.info("Starting audio playback ...")
    try:
        while not stop_event.is_set():
            # Adopt current connection in case of restart
            connection = globals().get("connection", initial_connection)
            raw_event = None
            try:
                raw_event = connection.recv()
            except Exception as _:
                time.sleep(0.05)
                continue

            if raw_event is None:
                continue

            try:
                event = json.loads(raw_event)
                event_type = event.get("type")
                print(f"Received event:", {event_type})

                if event_type == "session.created":
                    session = event.get("session")
                    logger.info(f"Session created: {session.get('id')}")
                    write_conversation_log(f"SessionID: {session.get('id')}")

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    user_text = event.get("transcript", "") or ""
                    user_transcript = f'User Input:\t{user_text}'
                    print(f'\n\t{user_transcript}\n')
                    write_conversation_log(user_transcript)

                    # ------ TOPICS ------
                    topic = match_topic(user_text)
                    if topic:
                        mode = topic.get("mode")
                        if mode == "local":
                            handle_topic_local(topic)
                            RESPONSE_ACTIVE = False
                            try:
                                connection.send(json.dumps({"type": "response.cancel", "event_id": ""}))
                            except Exception:
                                pass
                            continue

                        elif mode == "agent":
                            handle_topic_agent(connection, topic)
                            # let it flow

                        elif mode == "agent_then_shutdown":
                            say_then_shutdown(connection, topic.get("reply"))
                            # let it flow to receive audio events

                elif event_type in ("response.created", "response.output_audio.started"):
                    RESPONSE_ACTIVE = True

                elif event_type == "response.text.done":
                    agent_text = f'Agent Text Response:\t{event.get("text", "")}'
                    print(f'\n\t{agent_text}\n')
                    write_conversation_log(agent_text)

                elif event_type == "response.audio_transcript.done":
                    agent_audio = f'Agent Audio Response:\t{event.get("transcript", "")}'
                    print(f'\n\t{agent_audio}\n')
                    write_conversation_log(agent_audio)

                elif event_type == "response.audio.delta":
                    if event.get("item_id") != last_audio_item_id:
                        last_audio_item_id = event.get("item_id")
                    bytes_data = base64.b64decode(event.get("delta", ""))
                    if bytes_data:
                        logger.debug(f"Received audio data of length: {len(bytes_data)}")
                    audio_player.add_data(bytes_data)
                    RESPONSE_ACTIVE = True

                elif event_type == "input_audio_buffer.speech_started":
                    print("Speech started")
                    audio_player.stop()

                elif event_type in ("response.done","response.completed", "response.finished", "response.output_audio.stopped"):
                    RESPONSE_ACTIVE = False
                    if globals().get("PENDING_SHUTDOWN", False):
                        globals()["PENDING_SHUTDOWN"] = False
                        logger.info("Farewell TTS finished. Shutting down.")
                        stop_event.set()

                elif event_type == "error":
                    error_details = event.get("error", {})
                    error_code = str(error_details.get("code", "Unknown"))
                    error_message = error_details.get("message", "No message provided")

                    if error_code == "response_cancel_not_active":
                        logger.info("Cancel ignored: there was no active response.")
                        write_conversation_log(f"Ignored cancel: {error_message}")
                        continue

                    # Automatic retry if farewell collided with active response
                    if error_code == "conversation_already_has_active_response" and globals().get("PENDING_SHUTDOWN", False):
                        logger.info("There is an active response. Retrying farewell when free...")
                        _retry_farewell_when_idle(connection, globals().get("LAST_FAREWELL_TEXT", LAST_FAREWELL_TEXT), max_wait_s=3.0)
                        continue

                    raise ValueError(f"Error received: Code={error_code}, Message={error_message}")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON event: {e}")
                continue

    except Exception as e:
        logger.error(f"Error in audio playback: {e}")
    finally:
        audio_player.terminate()
        logger.info("Playback done.")

# =========================
# Live updates / restart
# =========================
def send_session_partial_update(connection: "VoiceLiveConnection", *,
                                temperature: float = None,
                                vad_threshold: float = None,
                                vad_silence_ms: int = None,
                                remove_fillers: bool = None):
    """
    Sends a partial session update for hot-swappable parameters (no restart required).
    """
    global current_session_cfg
    new_cfg = json.loads(json.dumps(current_session_cfg))

    if temperature is not None:
        new_cfg["voice"]["temperature"] = float(temperature)
    if vad_threshold is not None:
        new_cfg["turn_detection"]["threshold"] = float(vad_threshold)
    if vad_silence_ms is not None:
        new_cfg["turn_detection"]["silence_duration_ms"] = int(vad_silence_ms)
    if remove_fillers is not None:
        new_cfg["turn_detection"]["remove_filler_words"] = bool(remove_fillers)

    patch = {"type": "session.update", "session": new_cfg, "event_id": ""}
    connection.send(json.dumps(patch))
    write_conversation_log(f"Session Hot Update: {json.dumps(patch)}")
    current_session_cfg = new_cfg

def restart_connection_with_voice(new_voice_name: str):
    """
    Restarts the WebSocket connection to apply a new voice setting.
    Used when changing the TTS voice.
    """
    global current_session_cfg, RUNTIME
    if not new_voice_name:
        return
    with RESTART_LOCK:
        try:
            # 1) Close current connection (if any)
            old_conn = globals().get("connection")
            if old_conn:
                try:
                    old_conn.close()
                except Exception:
                    pass

            # 2) Update local config with new voice
            current_session_cfg["voice"]["name"] = new_voice_name

            # 3) Create new client and connection using RUNTIME context
            client = AzureVoiceLive(
                azure_endpoint=RUNTIME["endpoint"],
                api_version=RUNTIME["api_version"],
                token=RUNTIME["token"],
            )
            new_conn = client.connect(
                project_name=RUNTIME["project_name"],
                agent_id=RUNTIME["agent_id"],
                agent_access_token=RUNTIME["token"]
            )

            # 4) Publish new connection globally
            globals()["connection"] = new_conn
            try:
                connection_queue.get_nowait()
            except Exception:
                pass
            connection_queue.put(new_conn)

            # 5) Send full configuration to new WS
            session_update = {"type": "session.update", "session": current_session_cfg, "event_id": ""}
            new_conn.send(json.dumps(session_update))
            print(f"[Restart] Connection restored with voice: {new_voice_name}")
            write_conversation_log(f'Restarted with voice "{new_voice_name}": {json.dumps(session_update)}')

        except Exception as e:
            logger.error(f"Failed to restart with new voice: {e}")

# =========================
# Command UI
# =========================
def read_keyboard_and_quit() -> None:
    """
    Reads user commands from the keyboard and executes them.
    Supports voice change, temperature, VAD, silence, fillers, show, bye, and quit.
    """
    print("Commands: /voice <name>, /temp <0-1>, /vad <0-1>, /silence <ms>, /fillers on|off, /show, /bye, q")
    while not stop_event.is_set():
        try:
            user_input = input()
            if not user_input:
                continue
            if user_input.strip().lower() == 'q':
                print("Quitting the chat...")
                stop_event.set()
                break

            connection = globals().get("connection", None)
            if not connection:
                print("No connection available.")
                continue

            if user_input.startswith("/voice "):
                name = user_input[len("/voice "):].strip()
                if name not in AZURE_VOICES:
                    print(f"Notice: '{name}' is not in the local catalog (it may still work if your Agent supports it).")
                restart_connection_with_voice(name)
                print(f"✔ Voice applied (with restart): {name}")

            elif user_input.startswith("/temp "):
                try:
                    t = float(user_input[len("/temp "):].strip())
                    send_session_partial_update(connection, temperature=t)
                    print(f"✔ Temperature = {t}")
                except:
                    print("Format: /temp 0.7")

            elif user_input.startswith("/vad "):
                try:
                    v = float(user_input[len("/vad "):].strip())
                    send_session_partial_update(connection, vad_threshold=v)
                    print(f"✔ VAD threshold = {v}")
                except:
                    print("Format: /vad 0.28")

            elif user_input.startswith("/silence "):
                try:
                    ms = int(user_input[len("/silence "):].strip())
                    send_session_partial_update(connection, vad_silence_ms=ms)
                    print(f"✔ VAD silence_duration_ms = {ms} ms")
                except:
                    print("Format: /silence 220")

            elif user_input.startswith("/fillers "):
                val = user_input[len("/fillers "):].strip().lower()
                if val in ("on", "off"):
                    send_session_partial_update(connection, remove_fillers=(val == "on"))
                    print(f"✔ remove_filler_words = {val}")
                else:
                    print("Use: /fillers on|off")

            elif user_input.strip() == "/show":
                print("Voices in local catalog:")
                for v in AZURE_VOICES.keys():
                    print("  -", v)

            elif user_input.strip().lower() == "/bye":
                conn = globals().get("connection")
                if conn:
                    say_then_shutdown(conn, "It was a pleasure talking to you. See you later! 👋")
                else:
                    stop_event.set()

            else:
                print("Unrecognized command. Try /show")
        except EOFError:
            break
        except Exception as e:
            print(f"Error in command: {e}")

# =========================
# Utilities
# =========================
def write_conversation_log(message: str) -> None:
    """
    Writes a message to the conversation log file.
    """
    with open(f'logs/{logfilename}', 'a') as conversation_log:
        conversation_log.write(message + "\n")

# =========================
# Main
# =========================
def main() -> None:
    """
    Main entry point for the chat program.
    Initializes Azure credentials, sets up connections, threads, and starts the chat.
    """
    endpoint = os.environ.get("AZURE_VOICE_LIVE_ENDPOINT") or "<https://your-endpoint.azure.com/>"
    agent_id = os.environ.get("AI_FOUNDRY_AGENT_ID") or "<your-agent-id>"
    project_name = os.environ.get("AI_FOUNDRY_PROJECT_NAME") or "<your-project-name>"
    api_version = os.environ.get("AZURE_VOICE_LIVE_API_VERSION") or "2025-05-01-preview"
    api_key = os.environ.get("AZURE_VOICE_LIVE_API_KEY") or "<your-api-key>"

    # Token with DefaultAzureCredential (keyless)
    credential = DefaultAzureCredential()
    scopes = "https://ai.azure.com/.default"
    token = credential.get_token(scopes)

    # Save context for restarts
    RUNTIME["endpoint"] = endpoint
    RUNTIME["project_name"] = project_name
    RUNTIME["agent_id"] = agent_id
    RUNTIME["api_version"] = api_version
    RUNTIME["token"] = token.token

    client = AzureVoiceLive(
        azure_endpoint=endpoint,
        api_version=api_version,
        token=token.token,
        # api_key = api_key,
    )

    connection = client.connect(
        project_name=project_name,
        agent_id=agent_id,
        agent_access_token=token.token
    )

    # Expose connection
    globals()["connection"] = connection
    try:
        connection_queue.get_nowait()
    except Exception:
        pass
    connection_queue.put(connection)

    # Initial configuration (from .env)
    session_update = build_session_update_from_env()
    connection.send(json.dumps(session_update))
    print("Session created: ", json.dumps(session_update))
    write_conversation_log(f'Session Config: {json.dumps(session_update)}')

    # Threads
    send_thread = threading.Thread(target=listen_and_send_audio, args=(connection,))
    receive_thread = threading.Thread(target=receive_audio_and_playback, args=(connection,))
    keyboard_thread = threading.Thread(target=read_keyboard_and_quit)

    print("Starting the chat ...")
    send_thread.start()
    receive_thread.start()
    keyboard_thread.start()

    keyboard_thread.join()
    stop_event.set()
    send_thread.join(timeout=2)
    receive_thread.join(timeout=2)

    # Shutdown
    try:
        connection.close()
    except Exception:
        pass
    print("Chat done.")

# --- Bootstrap ---
if __name__ == "__main__":
    """
    Bootstrap code for running the chat program.
    Sets up logging, loads environment, and handles signals.
    """
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.exists('logs'):
            os.makedirs('logs')
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logfilename = f"{timestamp}_conversation.log"
        logging.basicConfig(
            filename=f'logs/{timestamp}_voicelive.log',
            filemode="w",
            level=logging.DEBUG,
            format='%(asctime)s:%(name)s:%(levelname)s:%(message)s'
        )
        load_dotenv("./.env", override=True)

        def signal_handler(signum, frame):
            print("\nReceived interrupt signal, shutting down...")
            stop_event.set()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        main()
    except Exception as e:
        print(f"Error: {e}")
        stop_event.set()
