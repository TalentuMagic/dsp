"""The following project is intended for DSP and NNGA courses and its aim is to classify audio files based on a Neural Network that will analyze the sound and output a answer based of what it <'heard'>. The metadata is provided by the UrbanSound8K project.\n
For more info, check the soundata library or the following GitHub link https://github.com/soundata/soundata#quick-example"""

# imports needed for the Deep Learning Audio Classification Project to work
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas
import seaborn
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

# the given input WAV file to be analysed
# person talking is identified as 'children_playing'
# audio_path = "../DSP/dsp/microphone-results.wav"
# a simulation of a bark done by a person is identified well
audio_path = "../DSP/microphone-results.wav"
metadata_path = "../DSP/dsp/Audio Classification/UrbanSound8K/metadata/UrbanSound8K.csv"
audioDataset_path = "../DSP/dsp/Audio Classification/UrbanSound8K/audio"

# loading the audio and plotting the waveform of the file
audio_data, sample_rate = librosa.load(audio_path)
librosa.display.waveshow(audio_data, sr=sample_rate)
plt.ylabel("Freq. Amplitude")
plt.show()
print("The sample rate of the given WAV file is:", sample_rate)

# loading the metadata CSV
metadata = pandas.read_csv(metadata_path)

# doing the Mel-Frequency Cepstral Coefficients
mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
# print(mfccs.shape)
# print(mfccs)


def features_extractor(file_path):
    # load the file (audio)
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    # we extract mfcc
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    # in order to find out scaled feature we do mean of transpose of value
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features


# passing through all the audio data an extracting their features in a list
extractedFeatures = list()
for index, row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(
        audioDataset_path), 'fold'+str(row["fold"])+'/', str(row["slice_file_name"]))
    finalClass_labels = row["class"]
    features_data = features_extractor(file_name)
    extractedFeatures.append([features_data, finalClass_labels])

# converting the features to a pandas DataFrame
extracted_features_df = pandas.DataFrame(
    extractedFeatures, columns=['feature', 'class'])
extracted_features_df.head()

# Split the dataset into independent and dependent datasets
# X = the features set ; y = the classes set
X = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())

# Label Encoding -> Label Encoder
labelencoder = LabelEncoder()
y = tf.keras.utils.to_categorical(labelencoder.fit_transform(y))
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

num_labels = y.shape[1]

# 1.The first layer has 100 neurons. Input shape is 40 according to the number of features with activation function as Relu (Rectified Linear Unit), and to avoid any overfitting, we'll use the Dropout layer at a rate of 0.5.
# 2.The second layer has 200 neurons with activation function as Relu and the drop out at a rate of 0.5.
# 3.The third layer again has 100 neurons with activation as Relu and the drop out at a rate of 0.5.
model = tf.keras.Sequential()
# first layer
model.add(tf.keras.layers.Dense(100, input_shape=(40,)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.5))
# second layer
model.add(tf.keras.layers.Dense(200))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.5))
# third layer
model.add(tf.keras.layers.Dense(100))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.5))
# final layer
# softmax converts a vector of values to a probability distribution
model.add(tf.keras.layers.Dense(num_labels))
model.add(tf.keras.layers.Activation('softmax'))

model.summary()

# compiling the model
# Compile defines the loss function, the optimizer and the metrics. We need the compiled model because training uses the loss function and the optimizer
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adam')
# training the model
# We will train a model for 100 epochs and batch size as 32. We use callback, which is a checkpoint to know how much time it took to train over data.
num_epochs = 100
num_batch_size = 32
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='./audio_classification.hdf5',
                                                  verbose=1, save_best_only=True)
start = datetime.now()
model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs,
          validation_data=(X_test, y_test), callbacks=[checkpointer], verbose='1')
duration = datetime.now() - start
print("Training completed in : ", duration)

# for TensorFlow ver.<=2.6
# model.predict_classes(X_test)

# for TensorFlow ver.>=2.6
predict_X = model.predict(X_test)
classes_X = np.argmax(predict_X, axis=1)
print(classes_X)

# working on the given audio file
mfccs_features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
# Reshape MFCC feature to 2-D array
mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
# predicted_label=model.predict_classes(mfccs_scaled_features)
X_predict = model.predict(mfccs_scaled_features)
predicted_label = np.argmax(X_predict, axis=1)
print(predicted_label)
prediction_class = labelencoder.inverse_transform(predicted_label)
print(prediction_class)
