
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV



tscv = TimeSeriesSplit(n_splits=5)

covid = pd.read_excel('chiffre_covid_14032021.xlsx',index_col='date',parse_dates=True)#,index_col='date',parse_dates=True)

covid = covid[covid['granularite']=='pays']

#print(covid.keys())
for i in covid:
    if not (covid[i].name in ["deces",'cas_confirmes','reanimation','hospitalises']):
        covid.drop([i], axis=1, inplace = True)
covid=covid[5:]

covid = covid.dropna(axis=0)

subject=['hospitalises','cas_confirmes','reanimation']
result=["deces"]

size=320

X = covid[:size][subject]
y = covid[:size][result]
y = y.values.reshape(size,)
X_result=covid[size:][subject]
#X_result=X_result.values.reshape(348-size,1)
y_result=covid[size:][result]
y_result = y_result.values.reshape(348-size,)

params={
    'mlpregressor__max_iter': [10000],
}

model=make_pipeline(StandardScaler(),MLPRegressor())


grid=GridSearchCV(model,param_grid=params,cv=tscv)#,scoring=scorer)


print(model.get_params())
grid.fit(X,y)
print(grid.score(X,y))
print(grid.score(X_result,y_result))
print(grid.best_params_)

plt.figure()
plt.plot(covid.index,covid['deces'])
plt.plot(covid.index[size:],grid.predict(X_result))
plt.plot(covid.index[:size],grid.predict(X))
plt.show()