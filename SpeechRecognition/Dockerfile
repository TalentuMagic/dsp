FROM python:3.11
WORKDIR /
RUN pip install --no-cache-dir -r openai-whisper SpeechRecognition pyaudio pocketsphinx
COPY . .
COPY ../microphone-results.wav .
CMD ["python", "whisper_recognition.py"]