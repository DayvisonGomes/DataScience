import numpy as np
from random import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

def Perceptron():
    x = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1]).T

    dim = x.shape[1]
    w = [2*random() -1 for i in range(dim)]
    b = 2*random() -1

    learning_rate = 0.15

    def step(x):
        if x > 0:
            return 1
        else:
            return 0

    for passo in range(101):
        custo = 0
        for x_n,y_n in zip(x,y):
            y_pred = np.dot(x_n,w) + b
            y_pred = step(y_pred)
            error = y_n - y_pred
            w = w + learning_rate*np.dot(error,x_n)
            b = b + learning_rate*error
            custo+= error**2

        if passo % 10 == 0:
            print('Passo {}: {}'.format(passo,custo))

    print('w:',w)
    print('b:',b)
    print('y_pred:',np.dot(x,w) + b)



def Perceptron_linear():

    df = pd.read_csv('medidas.csv')

    x = df.Altura.values.reshape(-1,1)
    y = df.Peso.values

    dim = x.shape[1]
    w = [2*random() -1 for i in range(dim)]
    b = 2*random() -1

    for passo in range(10001):
        custo = 0
        for x_n,y_n in zip(x,y):
            y_pred = np.dot(x_n,w) + b
            error = y_n - y_pred
            w = w + 1e-7*np.dot(error,x_n)
            b = b + 1e-2*error
            custo+= error**2

        if passo % 1000 == 0:
            print('Passo {}: {}'.format(passo,custo))

    print('w:',w)
    print('b:',b)
    #print('y_pred:',np.dot(x,w) + b)

    plt.title('Medidas')
    plt.plot(x,w*x+b,label='Função',color='darkred')
    plt.legend()
    plt.scatter(x,y)
    plt.xlabel('Altura')
    plt.ylabel('Peso')
    plt.show()
    
Perceptron()

def Perceptron_linear_():

    df = pd.read_csv('notas.csv')
   
    x = df[['prova1','prova2','prova3']].values
    y = df.final.values
    
    opa = MinMaxScaler(feature_range=(-1,1))
    x = opa.fit_transform(x.astype(np.float64))

    plt.subplot(1,3,1)
    plt.xlabel('Prova 1')
    plt.ylabel('Final')
    plt.scatter(x[:,0],y)
    
    plt.subplot(1,3,2)
    plt.xlabel('Prova 2')
    plt.ylabel('Final')
    plt.scatter(x[:,1],y)

    plt.subplot(1,3,3)
    plt.xlabel('Prova 3')
    plt.ylabel('Final')
    plt.scatter(x[:,2],y)
    
    plt.show()

    dim = x.shape[1]
    w = [2*random() -1 for i in range(dim)]
    b = 2*random() -1

    learning_rate = 1e-2
    for passo in range(2001):
        custo = 0
        for x_n,y_n in zip(x,y):
            y_pred = np.dot(x_n,w) + b
            error = y_n - y_pred
            w = w + learning_rate*np.dot(error,x_n)
            b = b + learning_rate*error
            custo+= error**2

        if passo % 200 == 0:
            print('Passo {}: {}'.format(passo,custo))

    print('w:',w)
    print('b:',b)
    #print('y_pred:',np.dot(x,w) + b)
    lm = LinearRegression()
    lm.fit(x,y)
    print(lm.coef_,lm.intercept_)
