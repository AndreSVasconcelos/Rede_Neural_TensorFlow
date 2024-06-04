# Importar bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Carregar o dataset
dataset = pd.read_csv('arquivos/personagens.csv')
#print(dataset.shape) # mostra o tamanho do dataset.shape
#print(dataset.head()) # mostra as 5 primeiras linhas
#print(dataset.tail()) # mostra as 5 ultimas linhas

# Graficos
#sns.countplot(x='classe', data=dataset)
#sns.heatmap(dataset.corr(numeric_only=True), annot=True)

X = dataset.iloc[:, 0:6].values # Selecionar as colunas 0 a 6
y = dataset.iloc[:, 6].values # Selecionar a coluna 6

y = (y == 'Bart') # Transformar Bart para True e Homer para False

# Dividir o dataset em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Construir rede Neural (6 entradas, 1 saida) (entradas + saidas)/2 = (6 + 1)/2 ~= 4 neuronios nas camadas ocultas 
# (optaremos por 3 camadas ocultas)
rede_neural = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=[6]),
    tf.keras.layers.Dense(units=4, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=4, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=4, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
])
# Verificar o modelo
rede_neural.summary()

# Treinar o modelo
rede_neural.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
historico = rede_neural.fit(X_train, y_train, epochs=100, validation_split=0.1)

# Avaliar o modelo
print(historico.history.keys()) # historico.history.keys()
plt.plot(historico.history['val_loss'])
plt.plot(historico.history['val_accuracy'])

# Fazer previsoes
y_pred = rede_neural.predict(X_test)
y_pred = (y_pred > 0.5)
#print(y_pred)
print(accuracy_score(y_pred, y_test))
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True)