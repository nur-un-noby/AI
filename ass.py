from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = build_cnn((32,32,3))

history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=64
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("CIFAR-10 Test Accuracy:", test_acc)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.savefig("accuracy_plot.png")
plt.show()
