import tensorflow as tf

data = tf.keras.datasets.fashion_mnist  # loads the pre built database into data

# separates the dataset in training and test sets
(training_images, training_labels), (test_images, test_labels) = data.load_data()


# changes the shape of the array so that tensorflow understands that the images are greyscale
# normalises the pixel values to lie between 0 and 1
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images/255.0

# changes the shape of the array so that tensorflow understands that the images are greyscale
# normalises the pixel values to lie between 0 and 1
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images/255.0


"""
    the structure of the model using CNNs
    the first line depicts the first convolution layer where 64 convolutions are learned with a size of 3x3
    the second layer where the feature volumes have their dimensions reduced
    after convolutions and max pooling the feature volume is flattened and put through a traditional neural network. 
"""
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


# compiles the model with and appropriate optimizer and loss function, and a metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

# fits the training images to their labels
model.fit(training_images, training_labels, epochs=50)

#evaluates the model on the test images
model.evaluate(test_images, test_labels)


"""
    the book continues to explore the model by using the 'model.summary' command
    to learn more about this go to page 39 - Exploring the Convolutional Network
"""
