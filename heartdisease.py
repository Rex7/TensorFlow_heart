import tensorflow as tf
import pandas as pd
import numpy as np
path = "C:\\Users\\Regis charles\\Downloads\\heart-disease-uci\\heart.csv"
training_set = pd.read_csv(path)
test_set = pd.read_csv(path)
predict_set = pd.read_csv("C:\\Users\\Regis charles\\Downloads\\heart-disease-uci\\predict.csv",)
training_set['thal'] = pd.Categorical(training_set['thal'])
training_set['thal'] = training_set.thal.cat.codes
target=training_set.pop("target")
test_set['thal'] = pd.Categorical(test_set['thal'])
test_set['thal'] = test_set.thal.cat.codes

dataset=tf.data.Dataset.from_tensor_slices((training_set.values,target.values))
train_dataset = dataset.shuffle(len(training_set)).batch(1)

datase_testt=tf.data.Dataset.from_tensor_slices((test_set.values,target.values))
test_dataset = datase_testt.shuffle(len(test_set)).batch(1)
# new format
predict_set['thal'] = pd.Categorical(predict_set['thal'])
predict_set['thal'] = predict_set.thal.cat.codes
target=predict_set.pop("target")

model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.fit(train_dataset,epochs=3)
loss,accuracy=model.evaluate(train_dataset)
print(loss,accuracy)
prediction=model.predict(predict_set)
print(prediction)






