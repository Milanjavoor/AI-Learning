import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(type(x_train))
plt.imshow(x_train[6],cmap='gray')
print("the actual value of the digit: {0}".format(y_train[6]))
print(x_test.shape)
x_train=x_train/255.0
x_test=x_test/255.0
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128,  input_shape=( (784 ,  )), activation="relu"))
model.add(tf.keras.layers.Dense(128 , activation= "relu"))
model.add(tf.keras.layers.Dense(10))
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
history=model.fit(x_train , y_train , epochs=33 , batch_size=32 , steps_per_epoch=200,
                  verbose=3 , validation_steps=50 , validation_split=0.2)
model.evaluate(x_test,y_test)
y_pred=model.predict(x_test)
print(y_test)
print(y_pred)
plt.plot(history.history["sparse_categorical_accuracy"])
plt.plot(history.history["val_sparse_categorical_accuracy"])
plt.title("Model accuracy plot")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train","Test"], loc="upper left")
plt.show
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss plot")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train","Test"],loc="upper left")
plt.show()