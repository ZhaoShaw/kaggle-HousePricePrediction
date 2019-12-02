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
    def __init__(self,n_cross,train_x_lst=[],train_y_lst=[],test_x_lst=[],test_y_lst=[],Y_mid,rg):
        self.n_cross = n_cross
        self.train_x_lst = train_x_lst
        self.train_y_lst = train_y_lst
        self.test_x_lst = test_x_lst
        self.test_y_lst = test_y_lst
        self.Y_mid = Y_mid
        self.rg = rg
    
    def split(self,X,Y,n_cross):
        kf = KFold(n_splits=n_cross)
        for train_index,test_index in kf.split(X):
            self.train_x_lst.append((train_index,X[train_index]))
            self.train_y_lst.append((train_index,Y[train_index]))
            self.test_x_lst.append((test_index,X[test_index]))
            self.test_y_lst.append((test_index,Y[test_index]))
            
    def fit_1(self,X,prilearner_lst):
        Y_features = []
        for i in range(len(prilearner_lst)):
            Y_temp = np.zeros((1,2))
            for j in range(self.n_cross):
                prilearner_lst[i].fit(self.train_x_lst[j][1],self.train_y_lst[j][1])
                Y_predvalue = prilearner_lst[i].predict(self.test_x_lst[j][1])
                Y_pred= np.concatenate((self.test_x_lst[j][0],Y_predvalue),axis=1)
                Y_temp = np.concatenate((Y_temp,Y_pred))
            Y_temp = np.delete(Y_temp,0,0)
            Y_features[i] = pd.DataFrame(Y_temp,index=Y_temp[:,0]).sort_index()
        self.Y_mid = pd.concat(Y_features,axis=1,ignore_index=True)
        
    def fit_2(self,Y):
        self.rg = LinearRegression()
        self.rg.fit(self.Y_mid,Y)
        
    def predict(self,X_test):
        return self.rg.predict(X_test)

                
                
                
                