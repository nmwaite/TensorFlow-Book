import tensorflow as tf
import urllib.request
import zipfile
import cv2
import numpy as np

#Urls for the zi[ files
url = 'https://storage.googleapis.com/learning-datasets/horse-or-human.zip'
validation_url = "https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip"

file_name = "horse-or-human.zip"
training_dir = 'horse-or-human\\training\\'
urllib.request.urlretrieve(url, file_name)  # downloads the zip file using the url

# code used to extract the zip file downloaded above
# zip_ref = zipfile.ZipFile(file_name, 'r')
# zip_ref.extractall(training_dir)
# zip_ref.close()


validation_file_name = "validation-horse-or-human.zip"
validation_dir = 'horse-or-human\\validation\\'
urllib.request.urlretrieve(validation_url, validation_file_name)  # downloads the zip file using the url

# code used to extract the zip file downloaded above
# zip_ref = zipfile.ZipFile(validation_file_name, 'r')
# zip_ref.extractall(validation_dir)
# zip_ref.close()



from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale=1/255)  # creates an ImageDataGenerator which is used for data augmentation

# specify this will generate training data, with some hyperparameters target_size, class_mode
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),
    class_mode='binary'
)

validation_datagen = ImageDataGenerator(rescale=1/255) # creates an ImageDataGenerator which is used for data augmentation

# specify this will generate validation data, with some hyperparameters target_size, class_mode
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size = (300,300),
    class_mode = 'binary'
)

# the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

"""
Compiles the model using binary crossentropy because this is a binary classification problem
the optimizer is RMSprop (Root Mean Squared propagation)
metric = accuracy
"""
model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              metrics=['accuracy'])

"""
fits the training images to thier labels using the training data image generator made above
epochs = 2
uses validation_data to test the model on a validation set top identify overfitting
"""
history = model.fit(
    train_generator,
    epochs=2,
    validation_data=validation_generator)

