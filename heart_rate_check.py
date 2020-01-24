import itertools as t

import tensorflow as tf
import pandas as pd

path = "C:\\Users\\Regis charles\\Downloads\\heart-disease-uci\\heart.csv"
columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
           "thal", "target"]
training_set = pd.read_csv(path,
                           skipinitialspace=True,
                           skiprows=1, names=columns)
test_set = pd.read_csv(path, skipinitialspace=True,
                       skiprows=1, names=columns)
predict_set = pd.read_csv("C:\\Users\\Regis charles\\Downloads\\heart-disease-uci\\predict.csv",
                          skipinitialspace=True,
                          skiprows=1, names=columns)
features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
            "thal"]
feature_colums = [tf.feature_column.numeric_column(k) for k in features]

estimator = tf.estimator.LinearRegressor(
    feature_columns=feature_colums,
    model_dir="train_heart"
)
boston = pd.DataFrame({k: training_set[k].values for k in features})
print(boston.isnull().count())


def input_func(data_set, num_epochs=None, n_batch_sizes=128, shuffle=True):
    return tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in features}),
        y=pd.Series(data_set["target"].values),
        num_epochs=num_epochs,
        batch_size=n_batch_sizes,
        shuffle=shuffle

    )


estimator.train(input_func(
    training_set, num_epochs=None, n_batch_sizes=128, shuffle=False), steps=1000)
eval = estimator.evaluate(input_func(test_set, 1, 128))
loss = eval["loss"]
print(eval)
print("Loss: {0:f}".format(loss))
predict_detail = estimator.predict(input_func(predict_set,
                                              num_epochs=1,
                                              shuffle=False))
predictions = list(p["predictions"]
                   for p in t.islice(predict_detail, 4
                                      ))
percentage=predictions[2].astype(float)
temp=percentage[0]*100
if(temp>90):
    print("sorry you have a heart condition check with a doctor")
    #print("percentage new percentage{}".format(percentage[0]*100))
    print("Formatted Number with percentage: "+"{:.2%}".format(percentage[0]))
else:
    print("you are healthy and you dont have a heart condition")
    print("Formatted Number with percentage: " + "{:.2%}".format(percentage[0]))


