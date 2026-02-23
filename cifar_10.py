import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np

# Load CIFAR-10
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten labels (because CIFAR gives shape (n,1))
y_train = y_train.flatten()
y_test = y_test.flatten()

# Build Model
model = Sequential()
model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Final Test Accuracy:", test_acc)

# Predict
y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# #plt.legend()
# plt.savefig ("/home/nurun-noby/Desktop/AI_assignment/accuracy_score_cifar.png")
# plt.show()

plt.figure(figsize=(10,10))

plt.figure(figsize=(10,10))
for i in range (36):
    plt.subplot(6,6,i+1)
    plt.imshow(X_test[i])
    plt.title(class_names[y_pred[i]])
    plt.axis ('off')
plt.tight_layout()
plt.savefig ("/home/nurun-noby/Desktop/AI_assignment/plot_ann_mnist_cifar.png")
plt.show()
