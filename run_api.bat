@echo off
echo Iniciando API de clasificación de dígitos MNIST...
call .\env_cnn\Scripts\activate
echo Entorno virtual activado
echo Ejecutando API en http://localhost:8000
python main.py
echo API detenida
pause