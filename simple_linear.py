import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,InputLayer

x=np.linspace(-10,10,1000)


# print(x)

def generate_y(x):
	return 5*x+10

y= generate_y(x)
y=y/100
Y=y.reshape(-1,1)
x= x/10
X=x.reshape(-1,1)

X_train, X_temp , Y_train, Y_temp = train_test_split(X,Y,test_size=0.3,random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

model = Sequential ([
		Dense(1,input_shape=(1,))
	])
model.compile(optimizer='adam',loss='mse',metrics=['mae'])

model.fit(X_train,Y_train,validation_data=(X_val,Y_val),epochs=80)

test_loss,test_mae = model.evaluate(X_test,Y_test)
print("Test loss:",test_loss)

y_pred = model.predict(X)

plt.figure()
plt.scatter(x, y, label='Original y')
plt.scatter(x, y_pred, label='Predicted y')
plt.legend()
plt.savefig("/home/nurun-noby/Desktop/AI_assignment/plot_linear.png")
print("plot saved")
plt.show()
