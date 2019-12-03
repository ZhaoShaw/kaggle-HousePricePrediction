# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 21:18:07 2019

@author: hasee
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

class StackingModel():
    def __init__(self,prilearner_lst,n_cross):
        self.prilearner_lst = prilearner_lst
        self.n_cross = n_cross
        self.rg = None

    def fit(self,X,Y):
        train_x_lst = []
        train_y_lst = []
        test_x_lst = []
        test_y_lst = []
        kf = KFold(n_splits=self.n_cross)
        for train_index,test_index in kf.split(X):
            train_x_lst.append((train_index,X[train_index]))
            train_y_lst.append((train_index,Y[train_index]))
            test_x_lst.append((test_index,X[test_index]))
            test_y_lst.append((test_index,Y[test_index]))
        
        Y_features = []
        for i in range(len(self.prilearner_lst)):
            Y_temp = np.zeros((1,2))
            for j in range(self.n_cross):
                self.prilearner_lst[i].fit(train_x_lst[j][1],train_y_lst[j][1])
                Y_predvalue = self.prilearner_lst[i].predict(test_x_lst[j][1])
                Y_pred= np.concatenate((test_x_lst[j][0].reshape(-1,1),Y_predvalue.reshape(-1,1)),axis=1)
                Y_temp = np.concatenate((Y_temp,Y_pred))
            Y_temp = np.delete(Y_temp,0,0)
            Y_features.append(pd.DataFrame(Y_temp,index=Y_temp[:,0],columns=['order','values']).sort_index()['values'])
        Y_mid = pd.concat(Y_features,axis=1,ignore_index=True).values
        
        self.rg = LinearRegression()
        self.rg.fit(Y_mid,Y)
        
    def predict(self,X_test):
        Y_temp = np.zeros((X_test.shape[0],1))
        for i in range(len(self.prilearner_lst)): 
            Y_predtest = self.prilearner_lst[i].predict(X_test)
            Y_temp = np.concatenate((Y_temp,Y_predtest.reshape(-1,1)),axis=1)
        Y_temp = np.delete(Y_temp,0,1)
        
        Y_final = self.rg.predict(Y_temp)
        
        return Y_final
