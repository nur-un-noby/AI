import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# Load CIFAR-10
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten labels
y_train = y_train.flatten()
y_test = y_test.flatten()

# CNN Model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=10)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print("CIFAR-10 Test Accuracy:", test_acc)

y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#plt.legend()
plt.savefig ("/home/nurun-noby/Desktop/AI_assignment/accuracy_score_cifar.png")
plt.show()

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