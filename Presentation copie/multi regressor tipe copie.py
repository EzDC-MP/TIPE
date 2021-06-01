import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.multioutput import RegressorChain
from sklearn.model_selection import GridSearchCV
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RANSACRegressor
scorer = make_scorer(mean_squared_error, greater_is_better=False)
from sklearn.linear_model import TheilSenRegressor

tscv = TimeSeriesSplit(n_splits=10)

covid = pd.read_excel('chiffre_covid_14032021.xlsx',index_col='date',parse_dates=True)#,index_col='date',parse_dates=True)

covid = covid[covid['granularite']=='pays']

for i in covid:
    if not (covid[i].name in ["deces",'cas_confirmes','reanimation','hospitalises']):
        covid.drop([i], axis=1, inplace = True)
covid=covid[5:]

covid = covid.dropna(axis=0)

subject=['hospitalises','cas_confirmes','reanimation']
result=["deces"]

size=255

X = covid[:size][subject]
y = covid[:size][result]
X_result=covid[size:][subject]
y_result=covid[size:][result]

y = y.values.reshape(size,)
y_result = y_result.values.reshape(348-size,)

params={
    'theilsenregressor__tol': [0.001],
    'theilsenregressor__random_state': [i for i in range(45)],
    'theilsenregressor__max_iter':  [90000],

}

model=make_pipeline(StandardScaler(),TheilSenRegressor())


grid=GridSearchCV(model,param_grid=params,cv=tscv)

grid.fit(X,y)


p=covid[subject]

plt.figure()
plt.plot(covid.index,covid['deces'])
plt.plot(covid.index[size:],grid.predict(X_result))
plt.show()
