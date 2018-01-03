# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

data=pd.read_csv("train.csv")
dataTest=pd.read_csv("test.csv")

colonnes_choisies=["LotArea","LotFrontage","Neighborhood","YearBuilt","1stFlrSF","2ndFlrSF","FullBath","BedroomAbvGr"
                   ,"TotRmsAbvGrd"]

X=data[colonnes_choisies]
y=data.SalePrice

Xtrain,Xcsv,ytrain,ycsv=train_test_split(X,y,random_state=0)

OHE_Xtrain=pd.get_dummies(Xtrain)
OHE_Xcsv=pd.get_dummies(Xcsv)
final_train,final_csv=OHE_Xtrain.align(OHE_Xcsv,join='left',axis=1)

imputed_Xtrain=final_train.copy()
imputed_Xcsv=final_csv.copy()

colonnes_avec_donnees_manquantes=[col for col in imputed_Xtrain.columns
                                  if imputed_Xtrain[col].isnull().any()]
for col in colonnes_avec_donnees_manquantes:
    imputed_Xtrain[col+"_donnee_manquantes"]=imputed_Xtrain[col].isnull()
    imputed_Xcsv[col+"_donnee_manquantes"]=imputed_Xcsv[col].isnull()

my_imputer=Imputer()
imputed_Xtrain=my_imputer.fit_transform(imputed_Xtrain)
imputed_Xcsv=my_imputer.fit_transform(imputed_Xcsv)

rand_model=RandomForestRegressor()
rand_model.fit(imputed_Xtrain,ytrain)
prediction_prix=rand_model.predict(imputed_Xcsv)

#on integre maintenant nos donnees de test
Xtest=dataTest[colonnes_choisies]
imputed_Xtest=Xtest.copy()
for col in colonnes_avec_donnees_manquantes:
    imputed_Xtest[col+"_donnee_manquantes"]=imputed_Xtest[col].isnull()

imputed_Xtest=my_imputer.fit_transform(imputed_Xtest)
prediction_finale=rand_model.predict(imputed_Xtest)
