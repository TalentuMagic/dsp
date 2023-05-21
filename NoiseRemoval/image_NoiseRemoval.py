import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
import asyncio
import cv2
import os
import pandas
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime
import tensorflow as tf

# start the stopwatch
start = datetime.now()

imageDataset_path = "../DSP-NNGA/dsp-nnga/NoiseRemoval/simple_images"
metadata_path = "../DSP-NNGA/dsp-nnga/NoiseRemoval/simple_images/metadata.csv"


# loading the metadata CSV
metadata = pandas.read_csv(metadata_path)

# colored noise


def add_noise_to_data(imageData, noise_factor):
    shape = imageData.shape
    batch_size, height, width, channels = shape

    # Compute the amount of noise pixels based on the global noise level
    noise_pixels = int(noise_factor * height * width * channels)

    # Generate random coordinates for noise pixels
    noise_coords = np.random.randint(
        0, height, size=noise_pixels), np.random.randint(0, width, size=noise_pixels)

    # Create a copy of the image data
    noisy_image_data = np.copy(imageData)

    # Add noise to the image data at the selected coordinates
    for i in range(batch_size):
        for c in range(channels):
            noise = np.random.normal(
                loc=0.0, scale=30.0, size=noise_pixels).astype(np.uint8)
            noisy_image_data[i, :, :, c][noise_coords] += noise

    return np.clip(noisy_image_data, 0.0, 255.0) / 255.0


def add_noise(image):
    # Compute the amount of noise pixels based on the global noise level
    noise_pixels = int(noise_factor * image.size)

    # Generate random coordinates for noise pixels
    rows, cols, _ = image.shape
    noise_coords = np.random.randint(
        0, rows, size=noise_pixels), np.random.randint(0, cols, size=noise_pixels)

    # Create a copy of the image
    noisy_image = np.copy(image)

    # Add noise to the image at the selected coordinates
    noisy_image[noise_coords] = np.random.randint(
        0, 256, size=(noise_pixels, 3))

    return noisy_image


def features_extractor(file_path: str):
    # load the file (image)
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imageData = np.array(image)
    return imageData


async def extractFeature(row):
    """Extract the Features Data and Class Labels and output them as a list - used for async threading"""
    # Creating a thread for each audio file path
    file_name = await asyncio.to_thread(os.path.join, os.path.abspath(imageDataset_path), str(row["fold"]), str(row["slice_file_name"]))
    # Getting the labels for each class
    finalClass_labels = row["class"]
    # Creating a thread for each feature extraction, for each threaded file
    features_data = await asyncio.to_thread(features_extractor, file_name)
    return [features_data, finalClass_labels]

extractedFeatures = list()


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
print("Image Dataset Features reading completed in : ", duration)
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

# Resize images to a consistent shape
target_shape = (250, 250)  # Specify the desired shape of the images
resized_images_features = []
resized_images_class = []
for image_data in extracted_features_df['feature']:
    image = cv2.resize(image_data, target_shape)
    resized_images_features.append(image)

# Split the dataset into independent and dependent datasets
# X = the features set ; y = the classes set
X = np.array(resized_images_features)
y = np.array(extracted_features_df['class'].values.tolist())

# Label Encoding -> Label Encoder
labelEncoder = LabelEncoder()
# converting the classes set to a binary matrix
y = tf.keras.utils.to_categorical(labelEncoder.fit_transform(y))
# Train Test Split of 25% test and 75% train
# a 42 random state to get different train-test sets - "life, universe and everything is 42"
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

X_train = X_train.reshape(len(X_train), 250, 250, 3)
X_test = X_test.reshape(len(X_test), 250, 250, 3)
# create a 1D matrix for with the binary data of the labels
num_labels = y.shape[1]

# add noise
noise_factor = 0.6
X_train_noisy = add_noise_to_data(X_train, noise_factor)
X_test_noisy = add_noise_to_data(X_test, noise_factor)
# reshape to the proper format
X_train_noisy = X_train_noisy.reshape(-1, 250, 250, 3)
X_test_noisy = X_test_noisy.reshape(-1, 250, 250, 3)

# randomly select input image
index = np.random.randint(len(X_test))
# plot the image
plt.imshow(X_test[index])
plt.show()
# randomly select noisy input image
index = np.random.randint(len(X_test))
# plot the image
plt.imshow(X_test_noisy[index])
plt.show()

model = tf.keras.Sequential()
# encoder Neural Network
# first layer
model.add(tf.keras.layers.Conv2D(
    64, 3, input_shape=(250, 250, 3), padding='same'))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.5))
# second layer
model.add(tf.keras.layers.MaxPooling2D(5, padding='same'))
model.add(tf.keras.layers.Dropout(0.5))
# third layer
model.add(tf.keras.layers.Conv2D(
    32, 3, padding='same'))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.5))
# fifth layer
model.add(tf.keras.layers.MaxPooling2D(5, padding='same'))
model.add(tf.keras.layers.Dropout(0.5))

# decoder Neural Network
# first layer
model.add(tf.keras.layers.Conv2D(
    32, 3, padding='same'))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.5))
# second layer
model.add(tf.keras.layers.UpSampling2D(5))
model.add(tf.keras.layers.Dropout(0.5))
# third layer
model.add(tf.keras.layers.Conv2D(
    64, 3, padding='same'))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.5))
# fourth layer
model.add(tf.keras.layers.UpSampling2D(5))
model.add(tf.keras.layers.Dropout(0.5))
# output layer
model.add(tf.keras.layers.Conv2D(
    1, 3, padding='same'))
model.add(tf.keras.layers.Activation('sigmoid'))
model.add(tf.keras.layers.Dropout(0.5))

model.summary()
model.compile(optimizer='adam', loss='mean_squared_error')

# the neural network is taken wholly 100 times in samples of 32 before updating the model
num_epochs = 128
num_batch_size = 32
# each time the model gets updated, the checkpointer updated the .hdf5 file (Hierarchical Data Format)
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='../DSP-NNGA/imageDenoising.hdf5',
                                                  verbose=1, save_best_only=True)
# fit the model according to the neural network and the configurations done above
model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs,
          validation_data=(X_test, y_test), callbacks=[checkpointer], verbose='1')

# predict the results from model (get compressed images)
predict_X_noisy = model.predict(X_test_noisy)
# output the prediction
predict_X = model.predict(X_test)
# stop the stopwatch to measure how much time it takes to fit the model on the given Train-Test datasets
duration = datetime.now() - start
print("Training completed in : ", duration)


def runImageDenoising(image_path: str):
    # load the file (image)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imageData = np.array(image)
    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(imageData.astype(np.uint8))
    plt.title('Original Image')
    plt.axis('off')

    # Add noise to the image
    image_noisy = add_noise(imageData)

    # Display the noisy image
    plt.subplot(1, 2, 2)
    plt.imshow(image_noisy)
    plt.title('Noisy Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    X_predict = model.predict(image_noisy)
    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image_noisy)
    plt.title('Noisy Image')
    plt.axis('off')

    # Add noise to the image
    image_noisy = add_noise(imageData)

    # Display the noisy image
    plt.subplot(1, 2, 2)
    plt.imshow(X_predict)
    plt.title('Denoised Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


runImageDenoising(
    "../DSP-NNGA/dsp-nnga/NoiseRemoval/simple_images/flowers/flowers_24d.jpeg")
