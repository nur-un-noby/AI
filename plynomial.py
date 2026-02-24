import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import InputLayer,Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace (-15,15,5000)

def gen_y(x):
	return 7*x**4 - 4*x**3 - x + 6

y = gen_y(x)

x=(x-x.mean())/x.std()
y=(y-y.mean())/y.std()

x= x.reshape(-1,1)
y=y.reshape(-1,1)

x_train,x_temp,y_train,y_temp = train_test_split(x,y,test_size=0.3)
x_test,x_val,y_test,y_val = train_test_split(x_temp,y_temp,test_size=0.5)

model = Sequential()
model.add(InputLayer(input_shape=(1,)))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='mse',metrics=['mae'])
history = model.fit (x_train,y_train,epochs=10)

y_pred = model.predict(y_test)

plt.figure()
plt.scatter(x_test,y_test,label='original y')
plt.scatter(x_test,y_pred,label='predict y')
plt.legend()
plt.show()

plt.figure()
plt.scatter(y_test,y_pred)
plt.xlabel("true value")
plt.ylabel("predict value")
plt.plot([y_test.min(),y_test.max()],
	[y_test.min(),y_test.max()])
plt.show()
