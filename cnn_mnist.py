import tensorflow 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Flatten,Dense,Conv2D,MaxPooling2D
from tensorflow.keras.models import Sequential

#load mnist data
(X_train,y_train),(X_test,y_test)=keras.datasets.mnist.load_data()

#normalize
X_train=X_train/255
X_test=X_test/255

#reshape
X_train = X_train.reshape (-1,28,28,1)
X_test = X_test.reshape (-1,28,28,1)

#cnn model

model = Sequential()
model.add (Conv2D(32, (3,3),activation='relu',input_shape=(28,28,1)))
model.add (MaxPooling2D((2,2)))
model.add (Conv2D(64,(3,3),activation='relu'))
model.add (MaxPooling2D((2,2)))
model.add (Flatten())
model.add (Dense(64,activation='relu'))
model.add (Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(X_train,y_train,validation_split=0.2,epochs=10)

test_loss,test_accuracy = model.evaluate(X_test,y_test)
print("Test loss:" ,test_loss)
print ("Final Accuracy:",test_accuracy)
y_prob = model.predict (X_test)
y_pred = y_prob.argmax(axis =1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.savefig("/home/nurun-noby/Desktop/AI_assignment/cnn_mnist_accuracy.png")
plt.show()

# plt.figure(figsize=(10,10))
# for i in range (36):
# 	plt.subplot(6,6,i+1)
# 	plt.imshow(X_test[i])
# 	plt.title(y_pred[i])
# 	plt.axis('off')
# plt.tight_layout()
# plt.savefig("/home/nurun-noby/Desktop/AI_assignment/cnn_mnist_accuracy.png")
# plt.show()