
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

jour_decale=10
covid = covid.dropna(axis=0)
print(covid.count())

print(covid["total_deces_hopital"][35])
print(covid["total_deces_hopital"][45])
timeshift_day(covid["total_deces_hopital"],-jour_decale)
print(covid["total_deces_hopital"][35])