import numpy as np
import matplotlib.pyplot as plt
from random import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

def sigmoid(z):
    return 1/(1+ np.exp(-z))

df = pd.read_csv('anuncios.csv')
opa = MinMaxScaler(feature_range=(-1,1))

x = df[['idade','salario']].values
y = df.comprou.values.reshape(-1,1)

x = opa.fit_transform(x.astype(np.float64))

dim = x.shape[1]

w = 2* np.random.random((1,dim)) -1
b = 2* np.random.random() -1

learning_rate = 0.01

for i in range(1001):
    
    z = np.dot(x,w.T) + b
    y_pred = sigmoid(z)
    error = y - y_pred 
    w = w + learning_rate*np.dot(error.T,x)
    b = b + learning_rate*error.sum()
            
    if i % 100 == 0:
        cost = np.mean(-y*np.log(y_pred) - (1-y)*np.log(1-y_pred))
        print('{} : {}'.format(i,cost))

print('w:',w)
print('b:',b)

coefs = pd.DataFrame({'Colunas':['idade','salario'] , 'coefs':[w[0,0],w[0,1]]  })
coefs.set_index('Colunas',inplace=True)
print(coefs)

lm = LogisticRegression(C=1e15)
lm2 = lm.fit(x,y.ravel())
ww = lm2.coef_
wb = lm2.intercept_
print(ww,wb)