# Exemplo de cassificação Linear:
import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 2.5, 3, 4]
y = [1, 4, 7, 9, 15]
plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 20])

plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 20]) # minimo e maximo de cada vetor

# A função unique () é usada para encontrar os elementos únicos de um array. Retorna os elementos únicos classificados de um array
# np.unique([0,1,2,0,2,3,4,3,0,4])
# -> array([0, 1, 2, 3, 4])
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.show()

####
# Classificacao linear usando a base da dados do Titanic
####

# Instalacao do sklearn e importacao dos modulos
!pip install -q sklearn
%tensorflow_version 2.x 
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

# Importacao dos dados - dados de passageiros do Titanic
# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# Analisando dos dados
# dftrain.head()
# dftrain.describe()
# dftrain.shape
# y_train.head()
# dftrain.age.hist(bins=20)
# dftrain.sex.value_counts().plot(kind='barh')
# dftrain['class'].value_counts().plot(kind='barh')
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')

# separando colunas numericas e categoricas
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # obtém uma lista de todos os valores únicos de determinada coluna
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

# Criando uma input function que converte o Dataframe em objeto
def make_input_fn(data_df, label_df, num_epochs=20, shuffle=True, batch_size=32):
  def input_function():  # inner function, que será retornada
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # criando o tf.data.Dataset object com seus labels
    if shuffle:
      ds = ds.shuffle(1000)  # randomize data
    ds = ds.batch(batch_size).repeat(num_epochs)  # divide 0 dataset em batches de 32 e repete o processo para o número de épocas
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # aqui sera chamda a input_function que foi retornado para obter um objeto de conjunto de dados que podemos alimentar o modelo
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
eval_input_fn

# Criando o modelo
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# Treinando o modelo
linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # obtém métricas/estatísticas do modelo testando dados de teste

clear_output()  # clears console output
#print(result['accuracy'])  # retorna dicionário de estatísticas sobre nosso modelo. Aumentando as epocas, aumenta o resultado.
print(result)

# predições individuais para não sobrevivente/sobrevivente - 'probabilities': array([0.27637616, 0.7236239 ]
result = list(linear_est.predict(eval_input_fn))
# print(result[0])
# print(result[1])
# print(result[2])
print(dfeval.loc[3]) # dados do passageiro
print(y_eval.loc[3]) # 0-não sobreviveu, 1-sobreviveu
print(result[3]['probabilities'][1]) # probabilidade de sobreviver do passageiro acima 

# Gráfico de previsões do modelo
pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
probs.plot(kind='hist', bins=20, title='predicted probabilities')


