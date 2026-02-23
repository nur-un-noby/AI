import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# import mnist dataset
(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()

# normalize
X_train = X_train/255
X_test = X_test/255

#adding flatten because data is 28x28 

model = Sequential()

model.add (Flatten(input_shape=(28,28)))
model.add (Dense(128,'relu'))
model.add (Dense(64,'relu'))
model.add (Dense(10,'softmax'))

#compile the model

model.compile(loss = 'sparse_categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])

#train the model

history = model.fit(X_train,y_train,validation_split=0.2,epochs=20)

#accuracy 

y_prob = model.predict (X_test)
y_pred = y_prob.argmax(axis =1)
accuracy=accuracy_score(y_test,y_pred)
print ("final accuracy:",accuracy)

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# #plt.legend()
# plt.savefig ("/home/nurun-noby/Desktop/AI_assignment/accuracy_score.png")
# plt.show()

plt.figure(figsize=(10,10))
for i in range (36):
	plt.subplot(6,6,i+1)
	plt.imshow(X_test[i])
	plt.title(y_pred[i])
	plt.axis ('off')
plt.tight_layout()
plt.savefig ("/home/nurun-noby/Desktop/AI_assignment/plot_mnist.png")
plt.show()