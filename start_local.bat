@echo off
title EVA Cloud Assistant (Local Test)
cd /d "%~dp0"
echo =======================================================
echo   INICIANDO EVA CLOUD ASSISTANT (FastAPI)
echo   IA: Google Gemini Flash
echo   Voz: ElevenLabs (Laura @ 16kHz PCM)
echo   STT: Groq Whisper
echo =======================================================
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
