@echo off
echo ===================================================
echo   RESTARTING RAG SYSTEM
echo ===================================================

echo [1/3] Killing existing processes...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM litellm.exe >nul 2>&1

echo [2/3] Starting LiteLLM Proxy (Background)...
start /B litellm --config llms/litellm_config.yaml > litellm.log 2>&1

echo [3/3] Starting RAG Server...
python -m src.server
