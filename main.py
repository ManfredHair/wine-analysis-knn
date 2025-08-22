# FREE WINE DATASET FROM
# Aeberhard, S. & Forina, M. (1992). Wine [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J.

import pandas as pd
import numpy as np
# from ucimlrepo import fetch_ucirepo
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# carregar dataset
wine = load_wine()

# criar estrutura dataframe
df = pd.DataFrame(wine.data, columns=wine.feature_names)
# adicionar targets no df, 0=Barolo , 1=Chianti , 2=Montepulciano
df['target'] = wine.target
df['target'] = df['target'].replace({0: 'Barolo', 1: 'Chianti', 2: 'Montepulciano'})
df.head()

df.describe()

x = wine.data
y = wine.target

#print(x.shape, y.shape)


### KNN com Holdout simples (70% treino - 30% teste)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

#print(x_train.shape, y_train.shape)
#print(x_test.shape, y_test.shape)

# Definir o modelo KNN com k=3
classifier1 = KNeighborsClassifier(n_neighbors=3)

# Treinar o modelo com os dados de treinamento
classifier1.fit(x_train, y_train)

# avaliar
y_pred = classifier1.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

###
