
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RANSACRegressor
scorer = make_scorer(mean_squared_error, greater_is_better=False)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV

### décale un matrice de données indéxées par un date à n jours en avance
def timeshift_day(data,n):
    data.index = data.index.shift(periods = n, freq ='D')


tscv = TimeSeriesSplit(n_splits=5)

covid = pd.read_csv('synthese-fra.csv',index_col='date',parse_dates=True)#,index_col='date',parse_dates=True)



print(covid.keys())
for i in covid:
    if not (covid[i].name in ["total_deces_hopital",'patients_reanimation',
                              'patients_hospitalises','total_cas_confirmes']):
        covid.drop([i], axis=1, inplace = True)

jour_decale=7
covid = covid.dropna(axis=0)
print(covid.count())
timeshift_day(covid["total_deces_hopital"],-jour_decale)


full_size=covid.count()[0]
print(full_size)
subject=['patients_reanimation',
         'patients_hospitalises','total_cas_confirmes']
result=["total_deces_hopital"]
#start:2020-03-17
predit_jour='2021-04-01'

X = covid[:predit_jour][subject]
y = covid[result][:predit_jour]
y = y.values.reshape(381,)
X_result=covid[predit_jour:][subject]
#X_result=X_result.values.reshape(348-size,1)
y_result=covid[result][predit_jour:]
y_result = y_result.values.reshape(40,)

params={
    'mlpregressor__max_iter': [10000],
    'mlpregressor__tol': [0.0001],
    'mlpregressor__n_iter_no_change': [2],
}

model=make_pipeline(StandardScaler(),MLPRegressor())


grid=GridSearchCV(model,param_grid=params,cv=tscv,scoring=scorer)


print(model.get_params())
grid.fit(X,y)
print(grid.score(X,y))
print(grid.score(X_result,y_result))
print(grid.best_params_)


timeshift_day(covid["total_deces_hopital"],jour_decale)
covid=covid[jour_decale:]


plt.figure()
plt.plot(covid.index,covid["total_deces_hopital"])
plt.plot(covid.index[374:],grid.predict(covid['2021-03-25':][subject]))
plt.plot(covid.index[:374],grid.predict(covid[:'2021-03-25'][subject]))
plt.show()