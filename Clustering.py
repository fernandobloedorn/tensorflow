# Previsão de temperatura de um determinado dia
# Importações
%tensorflow_version 2.x
!pip install tensorflow_probability==0.8.0rc0 --user --upgrade
import tensorflow_probability as tfp  # We are using a different module from tensorflow this time
import tensorflow as tf

# Modelo de um sistema meteorológico simples e tentar prever a temperatura em cada dia com as seguintes informações:
# 1. Os dias frios são codificados por 0 e os dias quentes são codificados por 1.
# 2. O primeiro dia de nossa sequência tem 80% de chance de estar frio.
# 3. Um dia frio tem 30% de chance de ser seguido por um dia quente.
# 4. Um dia quente tem 20% de chance de ser seguido por um dia frio.
# 5. Em cada dia, a temperatura é normalmente distribuída com média e desvio padrão 0 e 5 em um dia frio e média e desvio padrão 15 e 10 em um dia quente.

tfd = tfp.distributions  # criando atalho
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])  # referente ao item 2 acima
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5],
                                                 [0.2, 0.8]])  # referente aos itens 3 e 4 acima
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])  # referente ao item 5 acima

# criação do modelo de markov oculto
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=15)
# O número de steps representa o número de dias que gostaríamos de prever

mean = model.mean()

# due to the way TensorFlow works on a lower level we need to evaluate part of the graph
# from within a session to see the value of this tensor

# in the new version of tensorflow we need to use tf.compat.v1.Session() rather than just tf.Session()
with tf.compat.v1.Session() as sess:  
  print(mean.numpy())
