import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D,Flatten,MaxPooling2D,Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adagrad

(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()

odd_digits = [1,3,5,7,9]
train_mask = np.isin(y_train,odd_digits)
test_mask = np.isin(y_test,odd_digits)

x_train,y_train = x_train[train_mask],y_train[train_mask]
x_test,y_test = x_test[test_mask],y_test[test_mask]

x_train = x_train/255
x_test = x_test/255

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

y_train = np.array([odd_digits.index(i) for i in y_train])
y_test=np.array([odd_digits.index(i) for i in y_test])

model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(5,activation='softmax'))
model.summary()

optimizer = Adagrad(learning_rate = 0.003)
model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history1 =model.fit(x_train,y_train,epochs=10,batch_size=32)

for layer_name in ['conv2d','max_pooling2d','conv2d_1','max_pooling2d_1','conv2d_2','max_pooling2d_2']:
	model.get_layer(layer_name).trainable = False
model.summary()
model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=20,batch_size=32)

test_loss,test_accurcay = model.evaluate(x_test,y_test)
print ("Loss:",test_loss)
print("accuracy",test_accurcay)
y_prob= model.predict(x_test)
y_pred = y_prob.argmax(axis=1)

plt.figure()
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.show()

plt.figure(figsize=(6,6))
for i in range (36):
	plt.subplot(6,6,i+1)
	plt.imshow(x_test[i])
	plt.title(y_pred[i])
	plt.axis('off')
plt.tight_layout()
plt.show()