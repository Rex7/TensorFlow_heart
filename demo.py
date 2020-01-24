import itertools as t

import pandas as pd
import tensorflow as tf

path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]

training_Set = pd.read_csv("C:\\Users\\Regis charles\\Downloads\\boston_train\\boston_train.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
test_Set = pd.read_csv("C:\\Users\\Regis charles\\Downloads\\boston_train\\boston_test.csv", skipinitialspace=True,
                       skiprows=1, names=COLUMNS)
predict = pd.read_csv("C:\\Users\\Regis charles\\Downloads\\boston_train\\boston_predict.csv", skipinitialspace=True,
                      skiprows=1, names=COLUMNS)
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"
featureColums = [tf.feature_column.numeric_column(k) for k in FEATURES]
estimator = tf.estimator.LinearRegressor(
    feature_columns=featureColums,
    model_dir="train"
)
boston = pd.DataFrame({k: training_Set[k].values for k in FEATURES})
print(boston.isnull().sum())


def get_input_fun(data_set, num_epochs=None, n_batchSize=128, shuffle=True):
    return tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y=pd.Series(data_set[LABEL].values),
        batch_size=n_batchSize,
        num_epochs=num_epochs,
        shuffle=shuffle

    )


estimator.train(get_input_fun(
    training_Set, num_epochs=None, n_batchSize=128, shuffle=False), steps=1000)
ev = estimator.evaluate(get_input_fun(
    test_Set, num_epochs=1, n_batchSize=128, shuffle=False
))
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))
y = estimator.predict(
    input_fn=get_input_fun(predict,
                           num_epochs=1,
                           shuffle=False))
print(training_Set['medv'].describe())
predictions = list(p["predictions"]
                   for p in t.islice(y, 6))
print("Predictions: {}".format(str(predictions)))
