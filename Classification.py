# Classificação com Tensorflow
# Classificação de flores - Setosa, Versicolor e Virginica
# importações
%tensorflow_version 2.x  
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd

# Dataset
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

# Visualizando os dados
#train.head()
test.head()

train_y = train.pop('Species')
test_y = test.pop('Species')
train.head() 

train.shape 

# Input Function
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)
    
# Feature Columns
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

# Build a DNN (Deep Neural Network) with 2 hidden layers with 30 and 10 hidden nodes each.
# Two hidden layers of 30 and 10 nodes respectively.
# The model must choose between 3 classes.
classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, hidden_units=[30, 10], n_classes=3)

# Training
# Lambda to avoid creating an inner function previously
classifier.train(input_fn=lambda: input_fn(train, train_y, training=True), steps=5000)

# Avaliação
eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
#print(eval_result)
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# Função de predição
# Utilize esses exemplos para testar:
#               Setosa  | Versicolor | Virginica
# SepalLength |   5.1   |    5.9     |    6.9
# SepalWidth  |   3.3   |    3.0     |    3.1
# PetalLength |   1.7   |    4.2     |    5.4
# PetalWidth  |   0.5   |    1.5     |    2.1

def input_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
  valid = True
  while valid: 
    val = input(feature + ": ")
    if not val.isdigit(): valid = False

  predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))
        
