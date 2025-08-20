@echo off
echo 🤖 Ollama Setup for DocuVerse
echo ================================

echo 📥 Downloading and installing Ollama models...

REM Pull the model (this will download ~4GB)
ollama pull llama2

echo ✅ Ollama setup complete!
echo.
echo 🚀 Starting Ollama server...
ollama serve
