
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

jour_decale=15
covid = covid.dropna(axis=0)
print(covid.count())
timeshift_day(covid["total_deces_hopital"],-jour_decale)


full_size=covid.count()[0]
print(full_size)
subject=['patients_reanimation',
         'patients_hospitalises','total_cas_confirmes']
result=["total_deces_hopital"]
predit_jour=180
size=full_size-predit_jour

X = covid[:size][subject]
y = covid[:size][result]
y = y.values.reshape(size,)
X_result=covid[size:][subject]
#X_result=X_result.values.reshape(348-size,1)
y_result=covid[size:][result]
y_result = y_result.values.reshape(full_size-size,)

params={
    'mlpregressor__max_iter': [100000],
    'mlpregressor__tol': [0.001],
    'mlpregressor__n_iter_no_change': [4],
}

model=make_pipeline(StandardScaler(),MLPRegressor())


grid=GridSearchCV(model,param_grid=params,cv=tscv)


print(model.get_params())
grid.fit(X,y)
print(grid.score(X,y))
print(grid.score(X_result,y_result))
print(grid.best_params_)


timeshift_day(covid["total_deces_hopital"],jour_decale)
covid=covid[jour_decale:]


plt.figure()
plt.plot(covid.index,covid["total_deces_hopital"])
plt.plot(covid.index[size-jour_decale:],grid.predict(covid[size-jour_decale:][subject]))
plt.plot(covid.index[:size-jour_decale],grid.predict(covid[:size-jour_decale][subject]))
plt.show()