import pandas as pd
import tensorflow as tf
import numpy as np

path = "C:\\Users\\Regis charles\\Downloads\\heart-disease-uci\\heart.csv"
prediction_dataframe=pd.read_csv("C:\\Users\\Regis charles\\Downloads\\heart-disease-uci\\predict.csv")
prediction_set=prediction_dataframe.drop(columns="target")

training_dataframe=pd.read_csv(path)
print(training_dataframe.head())
training_set=training_dataframe.drop(columns="target")
training_set_label=tf.keras.utils.to_categorical(training_dataframe.target)

n_colums=training_set.shape[1]
print(n_colums)
model=tf.keras.Sequential()
model.add( tf.keras.layers.Dense(128,activation="relu",input_shape=(n_colums,)))
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dense(2,activation="softmax"))
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(training_set,training_set_label,epochs=15)
loss,accuracy=model.evaluate(training_set,training_set_label)
print(loss,accuracy)
prediction=model.predict(prediction_set)
print(np.argmax(prediction[2]))
