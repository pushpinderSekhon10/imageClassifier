import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import cv2 as cv

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()  # this is pulling from the
# cifar10 dataset which allows us to load this data into touples of training images and labels and testing images and labels
train_images, test_images = train_images / 255.0, test_images / 255.0  # this allows us to normalise the data and make the values of the images from 0-1 rather than 0-255

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
               'truck']  # these are the labels of the objects we will be looking for, and attempting to classify

for i in range(16):
    plt.subplot(4, 4, i + 1)  # in this we have a 4x4 grid and the i+1 is basically just moving through this grid
    plt.xticks([])
    plt.yticks([])  # we are not putting anything here as we do not need a coordinate system
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])

plt.show()


"""
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3),activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3),activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=25, validation_data=(test_images, test_labels))

loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

model.save('image_classifier.keras')
"""

model = models.load_model('image_classifier.keras')

img = cv.imread('lowResCar.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)
plt.show()

prediction = model.predict(np.array([img])/255.0)
idx = np.argmax(prediction)  # gets index of highest neuron
print('Predicted:', class_names[idx])