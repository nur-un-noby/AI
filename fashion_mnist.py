import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load Fashion-MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build model
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
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

# Predictions
y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# #plt.legend()
# plt.savefig ("/home/nurun-noby/Desktop/AI_assignment/accuracy_score_fashion.png")
# plt.show()

plt.figure(figsize=(10,10))
for i in range (36):
    plt.subplot(6,6,i+1)
    plt.imshow(X_test[i])
    plt.title(class_names[y_pred[i]])
    plt.axis ('off')
plt.tight_layout()
plt.savefig ("/home/nurun-noby/Desktop/AI_assignment/plot_mnist_fashion.png")
plt.show()