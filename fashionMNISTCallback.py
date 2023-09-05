import tensorflow as tf

"""
    A Class called myCallbacks which takes 'tf.keras.callbacks.Callback' as an argument
    within the class is a method which gives us details abot the logs of an epoch and
    based on the accuracy and a parameter determines if the training/fitting should stop
"""


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.95):
            print("\nReached 95% accuracy - STOPING")
            self.model.stop_training = True


# creates and instance of myCallback
callbacks = myCallback()


# loads the fashion_mnist db into mnist
mnist = tf.keras.datasets.fashion_mnist

# separates the mnist variable into a training and test set
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# normalising the imagesHorseHuman so their pixel values lie between 0 and 1
training_images = training_images/255.0
test_images = test_images/255.0

# creates the model with three layers
model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# compiles the model using the provided optimizer and loss function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""
    the fit method is different to the previous models
    and now contains a callbacks object, at the end of 
    each epoch the callback function will be called and
    if the requirements of the callback function are met 
    then training will stop 
"""
model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])