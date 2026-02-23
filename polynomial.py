import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import InputLayer, Dense



x = np.linspace(-10,10,1000)

def generate_y(x):
	return 3*x**2 + 5 * x + 10

x = x/10
X = x.reshape(-1,1)
y=generate_y(x)
y=y/100
Y = y.reshape(-1,1)

X_train,X_temp, Y_train, Y_temp = train_test_split(X,Y,test_size=0.3,random_state=42)
X_test, X_val, Y_test , Y_val = train_test_split(X_temp,Y_temp,test_size=0.5,random_state=42)

model = Sequential ([
	InputLayer(input_shape=(1,)),
	Dense(8,'relu'),
	Dense(1, )
	])
model.compile(
	optimizer = 'adam',
	loss = 'mse',
	metrics=['mae'])
model.fit(X_train,Y_train, validation_data=(X_val,Y_val),epochs=80)
test_loss, test_mae = model.evaluate (X_test,Y_test)
print("Test loss:" ,test_loss)

y_pred = model.predict(X)

plt.figure()
plt.scatter(x,y, label='original y')
plt.scatter(x,y_pred,label='predict y')
plt.legend()
plt.savefig("/home/nurun-noby/Desktop/AI_assignment/quardic.png")
print("saved png")
plt.show()