@echo off
cd /d c:\Users\jhonh\OneDrive\Escritorio\Clases_U\ia_2\minist-digit-classifier
call env_cnn\Scripts\activate
python -m app.realtime_with_api
pause