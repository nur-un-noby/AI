import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


(x_train,y_train),(x_test,y_test)= keras.datasets.mnist.load_data()

x_train=x_train/255
x_test=x_test/255

x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)

model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(5,activation='softmax'))
model.summary()

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train,epochs=10,validation_split=0.2)
test_loss,test_acc = model.evaluate(x_test,y_test)
print("test accuracy:",test_acc)

datagen = ImageDataGenerator(
	rotation_range=10,
	width_shift_range=0.1,
	height_shift_range=0.1)
datagen.fit(x_train)
model_aug= Sequential()
model_aug.add(Conv2D(32,(2,2),activation='relu',input_shape=(28,28,1)))
model_aug.add(MaxPooling2D(pool_size=(3,3),strides=(1,1)))
model_aug.add(Conv2D(64,(2,2),activation='relu'))
model_aug.add(MaxPooling2D(pool_size=(3,3),strides=(1,1)))
model_aug.add(Flatten())
model_aug.add(Dense(64,activation='relu'))
model_aug.add(Dense(10,activation='softmax'))
model_aug.summary()

model_aug.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history2=model_aug.fit(datagen.flow(x_train,y_train),batch_size=64,epochs=5,validation_data=(x_test,y_test))
aug_test_loss,aug_test_accuracy = model_aug.evaluate(x_test,y_test)
print("augmented accuracy:",aug_test_accuracy)

plt.figure()
plt.plot(history.history['accuracy'], label="Train Acc (Original)")
plt.plot(history.history['val_accuracy'], label="Val Acc (Original)")

plt.plot(history2.history['accuracy'], label="Train Acc (Augmented)")
plt.plot(history2.history['val_accuracy'], label="Val Acc (Augmented)")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Comparison")
plt.show()