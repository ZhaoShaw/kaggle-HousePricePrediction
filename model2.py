# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:53:35 2019

@author: hasee
"""
import numpy as np
import pandas as pd
from joblib import dump,load

def pred(feature):
    rg = load('./Stacking_Model.joblib')
    Y_pred = rg.predict(feature)
    return Y_pred
def main():
    feature = pd.read_csv('./testset.csv')
    Y_pred = pred(feature.values)
    df_pred = pd.DataFrame(Y_pred.reshape(-1,1),columns=['SalePrice'])
    df = pd.read_csv('./sample_submission.csv').drop(['SalePrice'],axis=1)
    df = pd.concat([df,df_pred],axis=1,join_axes=[df.index])
    df.to_csv('./sample_submission.csv',index=False)
#    print(df)
if __name__ == '__main__':
    main()