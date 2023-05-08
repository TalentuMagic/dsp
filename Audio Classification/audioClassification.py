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
import asyncio
import resampy

# start the stopwatch
start = datetime.now()

# the path to the metadata & audio dataset(s)
metadata_path = "../DSP/dsp/Audio Classification/UrbanSound8K/metadata/UrbanSound8K.csv"
audioDataset_path = "../DSP/dsp/Audio Classification/UrbanSound8K/audio"

# loading the metadata CSV
metadata = pandas.read_csv(metadata_path)


def features_extractor(file_path: str):
    """"The function takes the file path as the argument and extracts the audio data and sample rate of the audio file provided. Then does the MFCCs function to get the Features for the audio data and its sample rate using the Librosa library. The arithmetic mean is then computed using numpy to get the Scaled Features of the MFCC audio data."""
    # load the file (audio)
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    # we extract mfcc
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    # in order to find out scaled feature we do mean of transpose of value
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features


async def extractFeature(row):
    """Extract the Features Data and Class Labels and output them as a list - used for async threading"""
    # Creating a thread for each audio file path
    file_name = await asyncio.to_thread(os.path.join, os.path.abspath(audioDataset_path), 'fold'+str(row["fold"])+'/', str(row["slice_file_name"]))
    # Getting the labels for each class
    finalClass_labels = row["class"]
    # Creating a thread for each feature extraction, for each threaded file
    features_data = await asyncio.to_thread(features_extractor, file_name)
    return [features_data, finalClass_labels]

# Creating the list with all the extracted features for all audio files in the dataset
extractedFeatures = list()
# passing through all the audio data sequentially an extracting their features in a list
# for index, row in tqdm(metadata.iterrows()):
#     # taking each audio file provided in the dataset
#     file_name = os.path.join(os.path.abspath(
#         audioDataset_path), 'fold'+str(row["fold"])+'/', str(row["slice_file_name"]))
#     # selecting its 'class'
#     finalClass_labels = row["class"]
#     # extracting the Features of it
#     features_data = features_extractor(file_name)
#     # adding the features to the Features list
#     extractedFeatures.append([features_data, finalClass_labels])


async def extractedFeatures_func():
    """Passing through all the audio data using asyncio and extracting the futures for each row, appending to the extractedFeatures list as the output"""
    # Passing through all the audio data using asyncio
    global extractedFeatures
    tasks = []
    # Taking all the rows in the dataset
    for index, row in tqdm(metadata.iterrows()):
        # creating a task for each row and appending it to the tasks list
        task = asyncio.create_task(extractFeature(row))  # type: ignore
        tasks.append(task)
    # running the task in async and exporting the results to the extractedFeatures list
    extractedFeatures = await asyncio.gather(*tasks)
# running the async job
asyncio.run(extractedFeatures_func())

# stop the stopwatch to measure how much time it takes to fit the model on the given Train-Test datasets
duration = datetime.now() - start
print("Audio Dataset Features reading completed in : ", duration)
# To see where the work is done - currently using CPU only
ok = False
while (ok != True):
    debug = str(
        input("Do you wish to see the Log for Device Placement? (y/n)"+'\n'))
    if debug == 'y' or debug == 'Y':
        tf.debugging.set_log_device_placement(True)
        ok = True
    elif debug == 'n' or debug == 'N':
        tf.debugging.set_log_device_placement(False)
        ok = True
    elif (debug != 'y' or debug != 'Y') or (debug != 'n' or debug != 'N'):
        pass


# start the stopwatch
start = datetime.now()
# converting the features to a pandas DataFrame
extracted_features_df = pandas.DataFrame(
    extractedFeatures, columns=['feature', 'class'])
# get only the rows with the data
extracted_features_df.head()

# Split the dataset into independent and dependent datasets
# X = the features set ; y = the classes set
X = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())

# Label Encoding -> Label Encoder
labelEncoder = LabelEncoder()
# converting the classes set to a binary matrix
y = tf.keras.utils.to_categorical(labelEncoder.fit_transform(y))
# Train Test Split of 25% test and 75% train
# a 42 random state to get different train-test sets - "life, universe and everything is 42"
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# create a 1D matrix for with the binary data of the labels
num_labels = y.shape[1]

# Create the Neural Network on a Sequential Model to have exactly one tensor input and one tensor output
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

# outputs a summary of the neural network
model.summary()

# Compiling the model
# Compile defines the loss function, the optimizer and the metrics. We need the compiled model because training uses the loss function and the optimizer
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adam')
# training the model
# We will train a model for 100 epochs and batch size as 32. We use callback, which is a checkpoint to know how much time it took to train over data.
# the neural network is taken wholly 100 times in samples of 32 before updating the model
num_epochs = 100
num_batch_size = 32
# each time the model gets updated, the checkpointer updated the .hdf5 file (Hierarchical Data Format)
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='./DSP/dsp/audio_classification.hdf5',
                                                  verbose=1, save_best_only=True)
# fit the model according to the neural network and the configurations done above
model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs,
          validation_data=(X_test, y_test), callbacks=[checkpointer], verbose='1')

# for TensorFlow ver.<=2.6
# model.predict_classes(X_test)

# for TensorFlow ver.>=2.6
# output the prediction
predict_X = model.predict(X_test)
# output the class of the predicted response
classes_X = np.argmax(predict_X, axis=1)
print(classes_X)
# stop the stopwatch to measure how much time it takes to fit the model on the given Train-Test datasets
duration = datetime.now() - start
print("Training completed in : ", duration)


# the given input WAV file to be analysed
# person talking is identified as 'children_playing'
# audio_path = "../DSP/dsp/microphone-results.wav"
# a simulation of a bark done by a person is identified well
# audio_path = "../DSP/bark.wav"
# a street music recording is identified well
# audio_path = "../DSP/street.wav"
def runAudioClassification(audio_path: str):
    """The function runs the Sequential Keras Neural Network model on the given audio file's path given as argument.\n
    It outputs the prediction and the duration of the prediction."""
    # loading the audio and plotting the waveform of the file
    audio_data, sample_rate = librosa.load(audio_path)
    librosa.display.waveshow(audio_data, sr=sample_rate)
    plt.ylabel("Freq. Amplitude")
    plt.show()
    print("The sample rate of the given WAV file is:", sample_rate)
    plt.close()
    # doing the Mel-Frequency Cepstral Coefficients
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    # working on the given audio file
    # taking the MFCCs Features of the given audio file
    # print(mfccs.shape)
    # print(mfccs)
    mfccs_features = librosa.feature.mfcc(
        y=audio_data, sr=sample_rate, n_mfcc=40)
    # getting the Scaled Features
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    # Reshape MFCC feature to a 2D array
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

    # predicted_label=model.predict_classes(mfccs_scaled_features)
    # predict the feature of the Scaled Feature matrix using the trained model
    X_predict = model.predict(mfccs_scaled_features)
    # get the label for the predicted feature
    predicted_label = np.argmax(X_predict, axis=1)
    print(predicted_label)
    # get the class for the predicted label
    prediction_class = labelEncoder.inverse_transform(predicted_label)
    print(prediction_class)


# implementation to run the classification of different audio files given from the terminal with max 3 retries if bad selection
run = True
bad = 0
while (run):
    try:
        audio_path = '../DSP/' + \
            str(input("What is the name of the WAV file that you want to use?\n"))
    except:
        print("Wrong file path!\nTry again...\n")
        bad += 1
        continue
    runAudioClassification(audio_path=audio_path)
    print('\n')
    ask = str(
        input("Do you wish to continue with a different file? (y/n)"+'\n'))
    if ask == 'y' or ask == 'Y':
        bad = 0
        continue
    elif ask == 'n' or ask == 'N':
        break
    elif (ask != 'y' or ask != 'Y') or (ask != 'n' or ask != 'N') and bad != 3:
        print("Wrong selection!\nTry again...\n")
        bad += 1
    if bad == 3:
        print("Exiting...Too many wrong attempts.")
        break
