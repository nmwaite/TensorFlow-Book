import tensorflow as tf

# uses the keras in built dataset fashion_mnist
data = tf.keras.datasets.fashion_mnist

# uses data.load_data() to return the training and tests sets to the appropriate variable
(training_images, training_labels), (test_images, test_labels) = data.load_data()

"""
    normalising the pixel values in the imagesHorseHuman so that they lie between 0 and 1
    to find out more about normalisation go to:
    https://developers.google.com/machine-learning/data-prep/transform/normalization
"""
training_images = training_images/255.0
test_images = test_images/255.0

"""
    relu (Rectified Linear Unit) is an activation function, it returns a value if the result is greater than 0
    softmax is the final activation function that takes the result and determines which class the image belongs to 
"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # flattens in the 2D array input and transforms it into a 1D array
    tf.keras.layers.Dense(128, activation=tf.nn.relu),  # 128 is the number of neurons in this layer (hyperparameter)
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)  #
])


"""
    the loss function is sparse_categorical_crossentropy (used for classification)
    https://rmoklesur.medium.com/what-you-need-to-know-about-sparse-categorical-cross-entropy-9f07497e3a6f
    
    the optimizer is adam (an evolution of sgd, which is faster and more efficient)
    
    metrics allows for us to see the accuracy of the model as it is training
"""
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# fits the training imagesHorseHuman to the training labels
model.fit(training_images, training_labels, epochs=5)

# evaluates the model using the test set
model.evaluate(test_images, test_labels)


"""
    To explore the values of the output neurons go to page 29-30 of the book 
"""