import os
import io
import time
import json
import re
import datetime
import urllib.request
import ssl
from typing import Optional
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydub import AudioSegment
import static_ffmpeg

# Inicializar ffmpeg para procesamiento de audio en memoria
try:
    static_ffmpeg.add_paths()
except Exception as e:
    print(f"[FFMPEG] Info: {e}")

# Contexto SSL para llamadas seguras
ssl_ctx = ssl.create_default_context()

# --- Configuración de Entorno (API Keys leídas desde variables de Render) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "FGY2WhTYpPnrIDTdsKH5") # Laura (Español natural)

app = FastAPI(title="EVA Cloud Assistant", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Historial de conversación en memoria (últimas 8 interacciones)
conversation_history = []
MAX_HISTORY = 8

# Estado del Clima / Widget para la barra superior de EVA
widget_state = {
    "temp": "24°C",
    "icon": "SUN"
}

# --- 1. TRANSCRIPCIÓN CON GROQ WHISPER (STT Ultra-rápido) ---
def transcribe_audio_groq(pcm_bytes: bytes) -> str:
    """Convierte audio PCM 16kHz a WAV y lo transcribe con Groq Whisper en <0.2s."""
    if not GROQ_API_KEY:
        print("[STT ERROR] GROQ_API_KEY no configurada.")
        return ""
    try:
        audio = AudioSegment.from_raw(io.BytesIO(pcm_bytes), sample_width=2, frame_rate=16000, channels=1)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_bytes = wav_io.getvalue()
        
        boundary = "----WebKitFormBoundaryEVACloudSTT777"
        body = []
        body.append(f"--{boundary}".encode())
        body.append(b'Content-Disposition: form-data; name="model"')
        body.append(b'')
        body.append(b'whisper-large-v3-turbo')
        body.append(f"--{boundary}".encode())
        body.append(b'Content-Disposition: form-data; name="language"')
        body.append(b'')
        body.append(b'es')
        body.append(f"--{boundary}".encode())
        body.append(b'Content-Disposition: form-data; name="file"; filename="audio.wav"')
        body.append(b'Content-Type: audio/wav')
        body.append(b'')
        body.append(wav_bytes)
        body.append(f"--{boundary}--".encode())
        body.append(b'')
        payload = b"\r\n".join(body)

        req = urllib.request.Request("https://api.groq.com/openai/v1/audio/transcriptions", data=payload, headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "Mozilla/5.0"
        })
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=8) as r:
            res = json.loads(r.read().decode())
            return res.get("text", "").strip()
    except Exception as e:
        print(f"[STT ERROR] Falló transcripción con Groq: {e}")
        return ""

# --- 2. INTELIGENCIA Y EMOCIONES CON GOOGLE GEMINI FLASH ---
def generate_gemini_reply(user_prompt: str) -> tuple[str, str]:
    """Genera respuesta con Gemini Flash y extrae la emoción para la pantalla."""
    global conversation_history
    
    if not GEMINI_API_KEY:
        print("[GEMINI ERROR] GEMINI_API_KEY no configurada.")
        return "Hola, necesito que configures mi clave de Gemini en Render.", "SCEPTIC"
        
    system_prompt = (
        "Eres EVA, un robot asistente amigable, inteligente, tierna y carismática con forma física cúbica y pantalla de rostro. "
        "Responde siempre en español de forma muy concisa, natural y directa (máximo 1 a 2 oraciones breves). "
        "Sé expresiva y empática. Al final absoluto de cada respuesta, agrega exactamente una de las siguientes etiquetas de emoción entre corchetes: "
        "[HAPPY] para respuestas alegres o amables, [LOVE] para muestras de cariño o agradecimiento, [SURPRISED] para datos asombrosos o preguntas inesperadas, "
        "[SCEPTIC] para dudas o curiosidad, [ANGRY] para desacuerdos o errores, o [SAD] para disculpas o despedidas tristes."
    )

    contents = []
    for msg in conversation_history[-MAX_HISTORY:]:
        contents.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_prompt}]})

    gem_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={GEMINI_API_KEY}"
    payload = json.dumps({
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 150
        }
    }).encode("utf-8")

    try:
        req = urllib.request.Request(gem_url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=10) as r:
            res = json.loads(r.read().decode())
            raw_reply = res["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            emotion = "HAPPY"
            m_emo = re.search(r'\[(HAPPY|LOVE|SURPRISED|SCEPTIC|ANGRY|SAD)\]', raw_reply)
            if m_emo:
                emotion = m_emo.group(1)
                clean_reply = re.sub(r'\[(HAPPY|LOVE|SURPRISED|SCEPTIC|ANGRY|SAD)\]', '', raw_reply).strip()
            else:
                clean_reply = raw_reply

            conversation_history.append({"role": "user", "content": user_prompt})
            conversation_history.append({"role": "model", "content": clean_reply})
            if len(conversation_history) > MAX_HISTORY * 2:
                conversation_history = conversation_history[-MAX_HISTORY * 2:]

            return clean_reply, emotion
    except Exception as e:
        print(f"[GEMINI ERROR] {e}")
        return "Disculpa, tuve un microcorte en mis circuitos neuronales. ¿Me repites?", "SCEPTIC"

# --- 3. SÍNTESIS DE VOZ CON ELEVENLABS (Stream PCM 16kHz) ---
def stream_elevenlabs_audio(text: str):
    """Genera stream de audio PCM 16-bit 16kHz mono directo desde ElevenLabs."""
    if not ELEVENLABS_API_KEY:
        print("[ELEVEN ERROR] ELEVENLABS_API_KEY no configurada.")
        return
        
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream?output_format=pcm_16000"
    payload = json.dumps({
        "text": text,
        "model_id": "eleven_flash_v2_5",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }).encode("utf-8")

    req = urllib.request.Request(tts_url, data=tts_payload, headers={
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0"
    })
    
    with urllib.request.urlopen(req, context=ssl_ctx, timeout=12) as response:
        while True:
            chunk = response.read(1024)
            if not chunk:
                break
            yield chunk

# --- ENDPOINTS HTTP ---

@app.get("/")
@app.get("/health")
def health():
    return {"status": "online", "service": "EVA Cloud Assistant v3.0", "time": datetime.datetime.now().isoformat()}

@app.get("/widget_data")
def widget_data():
    """Retorna hora, fecha y clima actual para la barra superior de EVA."""
    now = datetime.datetime.now()
    dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    
    time_str = now.strftime("%H:%M")
    date_str = f"{dias[now.weekday()]} {now.day} {meses[now.month - 1]}"
    
    return JSONResponse({
        "time": time_str,
        "date": date_str,
        "temp": widget_state["temp"],
        "icon": widget_state["icon"]
    })

@app.get("/alert_poll")
def alert_poll():
    return Response(status_code=204)

@app.post("/audio")
async def handle_audio(request: Request):
    pcm_data = await request.body()
    if not pcm_data or len(pcm_data) < 1000:
        return Response(status_code=204)

    t0 = time.time()
    user_text = transcribe_audio_groq(pcm_data)
    t_stt = time.time()
    print(f"[STT] Usuario: \"{user_text}\" ({t_stt - t0:.2f}s)")

    if not user_text:
        return Response(status_code=204)

    reply_text, emotion = generate_gemini_reply(user_text)
    t_gem = time.time()
    print(f"[EVA ({emotion})] \"{reply_text}\" ({t_gem - t_stt:.2f}s)")

    headers = {
        "Content-Type": "application/octet-stream",
        "X-Eva-Emotion": emotion,
        "X-Keep-Listening": "false",
        "X-Eva-Card": "NONE"
    }

    return StreamingResponse(stream_elevenlabs_audio(reply_text), headers=headers)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
