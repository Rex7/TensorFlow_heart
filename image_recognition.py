import tensorflow as tf
import matplotlib.pyplot as pt
import numpy as np


minst = tf.keras.datasets.mnist

(train_x,train_label_y),(test_x,test_label_y)=minst.load_data()

print(train_label_y[1])
model =tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(train_x,train_label_y,epochs=3)
loss,accuracy=model.evaluate(test_x,test_label_y)
print(loss,accuracy)

prediction=model.predict(test_x)
print(np.argmax(prediction[1]))
pt.imshow(test_x[1])
pt.show()

