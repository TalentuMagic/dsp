"""The following project is intended for DSP and NNGA courses and its aim is to classify audio files based on a Neural Network that will analyze the sound and output a answer based of what it <'heard'>. The metadata is provided by the UrbanSound8K project.\n
For more info, check the soundata library or the following GitHub link https://github.com/soundata/soundata#quick-example"""
# imports needed for the Deep Learning Audio Classification Project to work
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas

# given input WAV file to be analysed
audio_path = "../DSP/dsp/microphone-results.wav"

data, sample_rate = librosa.load(audio_path)
librosa.display.waveshow(data, sr=sample_rate)
plt.ylabel("Freq. Amplitude")
plt.show()
print(sample_rate)
