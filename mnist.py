import random
import keras
from keras import layers
from keras import utils
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

size = 512
batch_size = 128
epochs = 20

train_size, test_size = 60000, 10000
width, height, classes = 28, 28, 10

x_train = x_train.reshape(train_size, width * height)
x_test = x_test.reshape(test_size, width * height)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = utils.to_categorical(y_train, classes)
y_test = utils.to_categorical(y_test, classes)

model = keras.models.Sequential([
    # layers.Flatten(input_shape=(width*height, )),
    layers.Dense(units=size, activation='relu', input_shape=(width*height,)),
    layers.Dropout(0.2),
    layers.Dense(units=size, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(units=classes, activation='softmax')])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

n = random.randint(0, 100)
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)

plt.imshow(x_test[n].reshape(28, 28), cmap='gray')
plt.title(f'Number: {np.argmax(res)}')
plt.show()
