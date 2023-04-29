FROM python:3.11
WORKDIR /
RUN pip install --no-cache-dir -r openai-whisper SpeechRecognition pyaudio pocketsphinx
COPY . .
CMD ["python", "whisper_recognition.py"]