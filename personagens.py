# Importar bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

# Carregar o dataset
dataset = pd.read_csv('arquivos/personagens.csv')
#print(dataset.shape) # mostra o tamanho do dataset.shape
#print(dataset.head()) # mostra as 5 primeiras linhas
#print(dataset.tail()) # mostra as 5 ultimas linhas

sns.countplot(x='classe', data=dataset)
sns.heatmap(dataset.corr(numeric_only=True), annot=True)