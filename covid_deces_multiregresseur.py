### Multi regresseur avec le modele SVR ###

import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import RegressorChain
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

##############################################################################
tscv = TimeSeriesSplit(n_splits=15) #Decoupage pour le CV adapté aux series temporelles.
scorer = make_scorer(mean_squared_error, greater_is_better=False) #https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error
model = make_pipeline(StandardScaler(), RegressorChain(SVR(), order=[0,2,1]))
params = {'regressorchain__base_estimator__C': [10**i for i in range(2,8)],
          'regressorchain__base_estimator__epsilon': [0.001,0.1,0.01],
          'regressorchain__base_estimator__kernel': ['rbf']}
          #'regressorchain__base_estimator__gamma' : ['scale', 'auto']}
grid = GridSearchCV(model, params, cv = tscv, verbose = 1)

##############################################################################
cdata = pd.read_csv('covid_numbers.csv',index_col='date',parse_dates=True) #nouvelle base de données màj (10-05-21)
#cdata = pd.read_csv('covid_numbers-10-05-21.csv',index_col='date',parse_dates=True)
cdata = cdata[cdata['granularite']=='pays']
for i in cdata:
    if not (cdata[i].name in ["deces", "reanimation", "cas_confirmes"]):
        cdata.drop([i], axis=1, inplace = True)
        
cdata = cdata.dropna(axis=0)

##############################################################################
date = '2020-11-01'

Y = cdata['2020-03-01':date]
x = pd.to_datetime(Y.index)

x = x.values.reshape(len(x),1)

grid.fit(x,Y)

x_futur = pd.to_datetime(cdata[date:].index)
x_futur = x_futur.values.reshape(len(x_futur),1)

cdata['deces'].plot(label="deces reels")
cdata['reanimation'].plot(label="reanimation reels")
cdata['cas_confirmes'].plot(label="cas confirmés réel")

plt.yscale('log')
plt.legend()
plt.plot(x_futur,grid.predict(x_futur),label="valeur predites")
plt.plot(x, grid.predict(x),label="valeur predites")

print(grid.best_params_)
