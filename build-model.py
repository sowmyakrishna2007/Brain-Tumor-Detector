import numpy as np
import tensorflow as tf
import cv2
import os

from sklearn.model_selection import train_test_split

EPOCHS = 30
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 2
TEST_SIZE = 0.4


def load_data():
    images = []
    labels = []
    for dirname, _, filenames in os.walk('brain_tumor_dataset'):
        for filename in filenames:
            image = cv2.imread(os.path.join(dirname, filename))
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            image = image / 255
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append(image)
            if "yes" in dirname:
                labels.append(0)
            elif "no" in dirname:
                labels.append(1)

    return images, labels


def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    images, labels = load_data()
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    model = get_model()
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=1)
    model.evaluate(x_test,  y_test, verbose=2)
    model.save("tumor_model.keras")


main()
