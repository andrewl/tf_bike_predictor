import tempfile
import pandas as pd
import tensorflow as tf

def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  #feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  feature_cols = {**continuous_cols, **categorical_cols}
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  print(continuous_cols.items())
  print(feature_cols)
  print(label)
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)


train_file = "./london_101.csv"
test_file = "./london_101_test.csv"
COLUMNS = ["day", "month", "minute", "status"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["day", "month"]
CONTINUOUS_COLUMNS = ["minute"]
df_all = pd.read_csv(train_file, usecols=COLUMNS, dtype={'day':str, 'month':str})
df_all[LABEL_COLUMN] = (df_all["status"].apply(lambda x: "bikes" in x)).astype(int)

df_train=df_all.sample(frac=0.8,random_state=200)
df_test=df_all.drop(df_train.index)

#define the feature columns
day = tf.contrib.layers.sparse_column_with_keys(
          column_name="day", keys=["0","1","2","3","4","5","6","7"])

month = tf.contrib.layers.sparse_column_with_keys(
          column_name="month", keys=["1","2","3","4","5","6","7","8","9","10","11","12"])

minute = tf.contrib.layers.real_valued_column("minute")

#create the model
model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns=[day, month, minute], model_dir=model_dir)

#train the model
m.fit(input_fn=train_input_fn, steps=200)

#test the training
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print(key, results[key])
