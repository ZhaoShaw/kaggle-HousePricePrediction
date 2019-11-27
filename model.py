# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 19:27:46 2019

@author: hasee
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from collections import Counter

import scipy.stats as ss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import Pipeline
from joblib import dump,load

def dataprocess(LotFrontage,LotArea,BsmtFinSF1,BsmtFinSF2,BsmtUnfSF,\
                TotalBsmtSF,FlrSF1st,FlrSF2nd,LowQualFinSF,GrLivArea,\
                BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,BedroomAbvGr,\
                KitchenAbvGr,TotRmsAbvGrd,Fireplaces,GarageCars,GarageArea,\
                WoodDeckSF,OpenPorchSF,EnclosedPorch,Porch3Ssn,ScreenPorch,\
                PoolArea,MiscVal,Flr_diff,FullBathNum,HalfBathNum,\
                YearBuilt,YearRemodAdd,GarageYrBlt,TimeSold,YearRemod_yn,\
                MSSubClass,MSZoning,Street,LotConfig,Neighborhood,\
                Condition1,Condition2,BldgType,HouseStyle,RoofStyle,\
                RoofMatl,Exterior1st,Exterior2nd,MasVnrType,Foundation,\
                Heating,CentralAir,Functional,GarageType,PavedDrive,\
                Fence,MiscFeature,Alley_nan,YearRemod_yr,\
                LotShape,LandContour,LandSlope,OverallQual,OverallCond,\
                ExterQual,ExterCond,BsmtQual,BsmtCond,BsmtExposure,\
                BsmtFinType1,BsmtFinType2,HeatingQC,Electrical,KitchenQual,\
                FireplaceQu,GarageFinish,GarageQual,GarageCond,PoolQC,SaleType,\
                SaleCondition,lower_d,ld_n):
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')
    train_labels = train['SalePrice']
    train = train.drop(['SalePrice'],axis = 1)
    train_test = pd.concat([train,test],axis=0,join='inner',ignore_index=True)
    
    '''deal with nan'''
    #use related feature to fillna
    train_test = related_fillna(train_test,['MSSubClass','Neighborhood'],'MSZoning')
    train_test = related_fillna(train_test,['Neighborhood'],'LotFrontage')
    train_test['Exterior1st'] = train_test['Exterior1st'].fillna(train_test['Exterior1st'].mode()[0])
    train_test['Exterior2nd'] = train_test['Exterior2nd'].fillna(train_test['Exterior2nd'].mode()[0])
    train_test = related_fillna(train_test,['Exterior1st','ExterQual'],'MasVnrType')
    #MasVnrType='None',MasVnrArea=0
    train_test = related_fillna(train_test,['MasVnrType'],'MasVnrArea') 
    train_test.loc[train_test['MasVnrType'] == 'None','MasVnrArea'] = 0
    
    # fillnan--Basement
    train_test['TotalBsmtSF'] = train_test['TotalBsmtSF'].fillna(0)
    train_test.loc[train_test['TotalBsmtSF'] == 0,['BsmtUnfSF','BsmtUnfSF','BsmtFinSF2','BsmtFinSF1']] = 0
    train_test.loc[train_test['TotalBsmtSF'] == 0,['BsmtFinType2','BsmtFinType1','BsmtQual','BsmtCond']] = 'None'
    train_test.loc[train_test['BsmtFinType2'].isnull(),'BsmtFinType2'] = train_test.loc[train_test['BsmtFinType2'].isnull(),'BsmtFinType1']
    train_test['BsmtQual'] = train_test['BsmtQual'].fillna(train_test['BsmtQual'].mode()[0])
    train_test['BsmtCond'] = train_test['BsmtCond'].fillna(train_test['BsmtCond'].mode()[0])
    train_test['BsmtExposure'] = train_test['BsmtExposure'].fillna('None')
    
    train_test['Electrical'] = train_test['Electrical'].fillna(train_test['Electrical'].mode()[0])
    train_test['BsmtFullBath'] = train_test['BsmtFullBath'].fillna(train_test['BsmtFullBath'].mode()[0])
    train_test['BsmtHalfBath'] = train_test['BsmtHalfBath'].fillna(train_test['BsmtHalfBath'].mode()[0])
    train_test = related_fillna(train_test,['OverallQual'],'KitchenQual')
    train_test['Functional'] = train_test['Functional'].fillna(train_test['Functional'].mode()[0])
    train_test['FireplaceQu'] = train_test['FireplaceQu'].fillna('None')
    
    # fillnan--Garage
    GarageCars_nan = train_test.loc[train_test['GarageType'] == 'Detchd','GarageCars'].mode()[0]
    GarageArea_nan = train_test[train_test['GarageType'] == 'Detchd'][train_test['GarageCars'] == GarageCars_nan]['GarageArea'].mode()[0]
    train_test['GarageCars'] = train_test['GarageCars'].fillna(GarageCars_nan)
    train_test['GarageArea'] = train_test['GarageArea'].fillna(GarageArea_nan)
    train_test.loc[train_test['GarageCars'] == 0,'GarageType'] = 'None'
    train_test.loc[train_test['GarageCars'] == 0,'GarageYrBlt'] = 'None'
    train_test.loc[train_test['GarageCars'] == 0,'GarageFinish'] = 'None'
    train_test.loc[train_test['GarageCars'] == 0,'GarageQual'] = 'None'
    train_test.loc[train_test['GarageCars'] == 0,'GarageCond'] = 'None'
    train_test = related_fillna(train_test,['YearRemodAdd','GarageType'],'GarageYrBlt')
    train_test = related_fillna(train_test,['GarageType','GarageYrBlt'],'GarageCond')
    train_test = related_fillna(train_test,['GarageType','GarageCond'],'GarageQual')
    train_test = related_fillna(train_test,['GarageType','GarageQual'],'GarageFinish')
    train_test['GarageYrBlt'] = train_test['GarageYrBlt'].replace('None',0)
    train_test['GarageYrBlt'] = train_test['GarageYrBlt'].replace(2207,2007)
    # fillnan--Pool
    train_test.loc[train_test['PoolArea'] == 0,'PoolQC'] = 'None'
    train_test = related_fillna(train_test,['OverallQual'],'PoolQC')
    
    train_test['Fence'] = train_test['Fence'].fillna('None')
    
    train_test.loc[train_test['MiscVal'] == 0,'MiscFeature'] = 'None'
    train_test['MiscFeature'] = train_test['MiscFeature'].fillna('Gar2')
    train_test['SaleType'] = train_test['SaleType'].fillna(train_test['SaleType'].mode()[0])
    
    ''' new features'''
    #use Alley_nan instead of Alley
    train_test['Alley_nan'] =  train_test['Alley'].apply(lambda x: 0 if pd.isnull(x) else 1)
    #use YearRemod_yr and YearRemod_yn
    train_test['YearRemod_yr'] = train_test['YearRemodAdd']-train_test['YearBuilt']
    train_test['YearRemod_yn'] = train_test['YearRemod_yr'].apply(lambda x: 1 if x !=0 else 0)
    #use abs(1stFlrSF-2ndFlrSF) 
    train_test['Flr_diff'] = abs(train_test['1stFlrSF'] - train_test['2ndFlrSF'])
    train_test['FullBathNum'] = train_test['BsmtFullBath'] + train_test['FullBath']
    train_test['HalfBathNum'] = train_test['BsmtHalfBath'] + train_test['HalfBath']
    train_test['TimeSold'] = train_test['YrSold'] + train_test['MoSold']
    #drop useless features
    train_test = train_test.drop(['Id','Alley','Utilities','YrSold','MoSold','most_count'],axis=1)
#    print(train_test)
    ''' scaling/encoder/lower decomposition'''
    # 'Condition1','Condition2','Exterior1st','Exterior2nd','OverallQual','OverallCond',\
    # 'MasVnrType','MasVnrArea','ExterQual','ExterCond','BsmtQual','BsmtCond'
    type_lst_col = ['MSSubClass','MSZoning','Street','LotConfig','Neighborhood',\
                'Condition1','Condition2','BldgType','HouseStyle','RoofStyle',\
                'RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',\
                'Heating','CentralAir','Functional','GarageType','PavedDrive',\
                'Fence','MiscFeature','Alley_nan','YearRemod_yr','SaleType',\
                'SaleCondition']
    type_lst = [MSSubClass,MSZoning,Street,LotConfig,Neighborhood,\
                Condition1,Condition2,BldgType,HouseStyle,RoofStyle,\
                RoofMatl,Exterior1st,Exterior2nd,MasVnrType,Foundation,\
                Heating,CentralAir,Functional,GarageType,PavedDrive,\
                Fence,MiscFeature,Alley_nan,YearRemod_yr,SaleType,\
                SaleCondition]
    seq_lst_col = ['LotShape','LandContour','LandSlope','OverallQual','OverallCond',\
               'ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure',\
               'BsmtFinType1','BsmtFinType2','HeatingQC','Electrical','KitchenQual',\
               'FireplaceQu','GarageFinish','GarageQual','GarageCond','PoolQC']
    seq_lst = [LotShape,LandContour,LandSlope,OverallQual,OverallCond,\
               ExterQual,ExterCond,BsmtQual,BsmtCond,BsmtExposure,\
               BsmtFinType1,BsmtFinType2,HeatingQC,Electrical,KitchenQual,\
               FireplaceQu,GarageFinish,GarageQual,GarageCond,PoolQC]
    #area/distance.etc
    num_lst_col = ['LotFrontage','LotArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',\
               'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',\
               'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',\
               'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea',\
               'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch',\
               'PoolArea','MiscVal','Flr_diff','FullBathNum','HalfBathNum',\
               'YearBuilt','YearRemodAdd','GarageYrBlt','TimeSold', 'YearRemod_yn']
    num_lst = [LotFrontage,LotArea,BsmtFinSF1,BsmtFinSF2,BsmtUnfSF,\
               TotalBsmtSF,FlrSF1st,FlrSF2nd,LowQualFinSF,GrLivArea,\
               BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,BedroomAbvGr,\
               KitchenAbvGr,TotRmsAbvGrd,Fireplaces,GarageCars,GarageArea,\
               WoodDeckSF,OpenPorchSF,EnclosedPorch,Porch3Ssn,ScreenPorch,\
               PoolArea,MiscVal,Flr_diff,FullBathNum,HalfBathNum,\
               YearBuilt,YearRemodAdd,GarageYrBlt,TimeSold,YearRemod_yn]
    type_lst_col.extend(seq_lst_col)
    type_lst.extend(seq_lst)
    for i in range(len(num_lst_col)):
        if not num_lst[i]:
            train_test[num_lst_col[i]] = MinMaxScaler().fit_transform(train_test[num_lst_col[i]].values.reshape(-1,1))
        else:
            train_test[num_lst_col[i]] = StandardScaler().fit_transform(train_test[num_lst_col[i]].values.reshape(-1,1))
    for i in range(len(type_lst_col)):
        if not type_lst[i]:
            train_test[type_lst_col[i]] = LabelEncoder().fit_transform(train_test[type_lst_col[i]])
            train_test[type_lst_col[i]] = StandardScaler().fit_transform(train_test[type_lst_col[i]].values.reshape(-1,1))
        else:
            train_test = pd.get_dummies(train_test,columns=[type_lst_col[i]])
    if lower_d:
        mypca = PCA(n_components=ld_n)
        features = mypca.fit_transform(train_test.values)
        print(mypca.explained_variance_ratio_)
        return features,train_labels
    return train_test,train_labels

def modeling(features,labels):
    X_tt,X_val,Y_tt,Y_val = train_test_split(features,labels,test_size=0.2)
    X_train,X_test,Y_train,Y_test = train_test_split(X_tt,Y_tt,test_size=0.2)
    
    models=[]
    models.append(('LinearRegression',LinearRegression()))
    models.append(('LogisticRegression',LogisticRegression()))
    models.append(('GBR',GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=6)))
    
    for rgname,rg in models:
        rg.fit(X_train,Y_train)
        xy_lst = [(X_train,Y_train),(X_test,Y_test),(X_val,Y_val)]
        
        for i in range(len(xy_lst)):
            x_part = xy_lst[i][0]
            y_part = xy_lst[i][1]
            Y_pred = rg.predict(x_part)
            print(i)
            print(rgname,'r2',r2_score(y_part,Y_pred))
            print(rgname,'MSE',mean_squared_error(y_part,Y_pred))
            
            dump(rg,'%s.joblib'%rgname)
            
def related_fillna(df_nan,relate_lst,col_nan):
    try:
        if df_nan[col_nan].dtype == 'int' or df_nan[col_nan].dtype == 'float':
            d = dict(df_nan.groupby(relate_lst)[col_nan].mean())
            b = df_nan[df_nan[col_nan].isnull()].index.get_values()
            for i in b:
                if len(relate_lst) >= 2:
                    df_nan.loc[df_nan.index[i],col_nan] = d[tuple(df_nan.loc[df_nan.index[i],relate_lst])]
                else:
                    df_nan.loc[df_nan.index[i],col_nan] = d[df_nan.loc[df_nan.index[i],relate_lst].values[0]]
            return df_nan 
        else:
            df_nan['most_count'] = df_nan.groupby(relate_lst)[col_nan].transform(lambda x: Counter(x).most_common(1)[0][0])
            d = dict(df_nan.groupby(relate_lst)['most_count'].first())
            b = df_nan[df_nan[col_nan].isnull()].index.get_values()
            for i in b:
                if len(relate_lst) >= 2:
                    df_nan.loc[df_nan.index[i],col_nan] = d[tuple(df_nan.loc[df_nan.index[i],relate_lst])]
                else:
                    df_nan.loc[df_nan.index[i],col_nan] = d[df_nan.loc[df_nan.index[i],relate_lst].values[0]]
            return df_nan
    except:
        print('datatype error')
    
def main():
    features,labels = dataprocess(LotFrontage=True,LotArea=True,BsmtFinSF1=True,BsmtFinSF2=True,BsmtUnfSF=True,\
                TotalBsmtSF=True,FlrSF1st=True,FlrSF2nd=True,LowQualFinSF=True,GrLivArea=True,\
                BsmtFullBath=True,BsmtHalfBath=True,FullBath=True,HalfBath=True,BedroomAbvGr=True,\
                KitchenAbvGr=True,TotRmsAbvGrd=True,Fireplaces=True,GarageCars=True,GarageArea=True,\
                WoodDeckSF=True,OpenPorchSF=True,EnclosedPorch=True,Porch3Ssn=True,ScreenPorch=True,\
                PoolArea=True,MiscVal=True,Flr_diff=True,FullBathNum=True,HalfBathNum=True,\
                YearBuilt=True,YearRemodAdd=True,GarageYrBlt=True,TimeSold=True,YearRemod_yn=True,\
                MSSubClass=True,MSZoning=True,Street=True,LotConfig=True,Neighborhood=True,\
                Condition1=True,Condition2=True,BldgType=True,HouseStyle=True,RoofStyle=True,\
                RoofMatl=True,Exterior1st=True,Exterior2nd=True,MasVnrType=True,Foundation=True,\
                Heating=True,CentralAir=True,Functional=True,GarageType=True,PavedDrive=True,\
                Fence=True,MiscFeature=True,Alley_nan=True,YearRemod_yr=True,\
                LotShape=False,LandContour=False,LandSlope=False,OverallQual=False,OverallCond=False,\
                ExterQual=False,ExterCond=False,BsmtQual=False,BsmtCond=False,BsmtExposure=False,\
                BsmtFinType1=False,BsmtFinType2=False,HeatingQC=False,Electrical=False,KitchenQual=False,\
                FireplaceQu=False,GarageFinish=False,GarageQual=False,GarageCond=False,PoolQC=False,\
                SaleType=False,SaleCondition=False,lower_d=True,ld_n=20)
    trainset_X = features[:1460,:]
    trainset_Y = labels.values.reshape(-1,1)
    testset = features[1460:,:]
    testset_df = pd.DataFrame(testset)
    testset_df.to_csv('./testset.csv',index=False)
    modeling(trainset_X,trainset_Y)
#    print(features,labels)
if __name__ == '__main__':
    main()