
import pandas as pd
from sklearn.model_selection import train_test_split
from models.EntityEmbeddingModel import EntityEmbedding
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def label(data, column):
    unique = data[column].unique()
    k = 0
    for str in unique:
        data.loc[data[column] == str, column] = k
        k += 1


def replace(data, weights, features, drop=True):
    for feature in features:
        data = data.merge(weights[feature], how='left', on=[feature])
        if drop == True:
            data = data.drop([feature], axis=1)
    return data


dataset = pd.read_csv('data/airline.csv')
dataset = dataset[:1000]

print(dataset)
label(dataset, 'Airline')
label(dataset, 'AirportFrom')
label(dataset, 'AirportTo')

dataset = dataset.drop(['Flight', 'Time'], axis=1)

model = EntityEmbedding()
model.add('Airline', input_shape=18, output_shape=8)
model.add('AirportFrom', input_shape=220, output_shape=10)
model.add('AirportTo', input_shape=71, output_shape=10)
model.add('DayOfWeek', input_shape=7, output_shape=5)
model.dense('Length', output_shape=1)
model.concatenate()

X = dataset.loc[:, ['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek', 'Length']]
y = dataset[['Delay']]

X['Airline'] = X['Airline'].astype(float, errors='raise')
X['AirportFrom'] = X['AirportFrom'].astype(np.float32)
X['AirportTo'] = X['AirportTo'].astype(np.float32)
X['DayOfWeek'] = X['DayOfWeek'].astype(np.float32)

X_train, X_ee, y_train, y_ee = train_test_split(X, y, test_size=0.25, random_state=44)

model.fit(X_ee, y_ee['Delay'], X_train, y_train['Delay'], epochs=12)

weights = model.get_weight()
X_train = replace(X_train, weights, model.embeddings)
X_ee = replace(X_ee, weights, model.embeddings)

print(X_train)
print(X_ee)

regressionModel = LogisticRegression()
regressionModel.fit(X_train, y_train['Delay'])

y_predict = regressionModel.predict(X_ee)

print(accuracy_score(y_ee['Delay'], y_predict))












