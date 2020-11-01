import tensorflow as tf
from tensorflow import keras


# GRADED FUNCTION: train_mnist
def train_mnist():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            # print(logs)
            if (logs.get('accuracy') >= 0.998):
                print("\nReached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()

    # YOUR CODE SHOULD END HERE

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    # YOUR CODE SHOULD START HERE

    training_images = training_images.reshape(60000, 28, 28, 1)
    test_images = test_images.reshape(10000, 28, 28, 1)

    training_images = training_images / 255.0
    test_images = test_images / 255.0

    # YOUR CODE SHOULD END HERE
    model = tf.keras.models.Sequential([
        # YOUR CODE SHOULD START HERE

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)

        # YOUR CODE SHOULD END HERE
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model fitting
    history = model.fit(  # YOUR CODE SHOULD START HERE
        training_images, training_labels, epochs=20, callbacks=[callbacks]
        # YOUR CODE SHOULD END HERE
    )
    # model fitting
    return history.epoch, history.history['accuracy'][-1]

train_mnist()
