from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import RegressorChain
from sklearn.model_selection import GridSearchCV
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
scorer = make_scorer(mean_squared_error, greater_is_better=False)

covid = pd.read_excel('chiffre_covid_14032021.xlsx',index_col='date',parse_dates=True)#,index_col='date',parse_dates=True)

covid = covid[covid['granularite']=='pays']

#print(covid.keys())
for i in covid:
    if not (covid[i].name in ["deces",'cas_confirmes','reanimation','hospitalises']):
        covid.drop([i], axis=1, inplace = True)
covid=covid[5:]

covid = covid.dropna(axis=0)

subject=[]
result=['hospitalises',"deces",'reanimation']

debut=30
size=278

X = covid.index[debut:size]
X = X.values.reshape(size-debut,1)
y = covid[debut:size][result]
X_result=covid.index[size:]
X_result=X_result.values.reshape(348-size,1)
y_result=covid[size:][result]



params={#'regressorchain__base_estimator__tol':[i*0.001 for i in range(1,20)],
        'regressorchain__base_estimator__epsilon': [0.0001],
        'regressorchain__base_estimator__C': [i*5000 for i in range (1,31)],
        #'regressorchain__base_estimator__cache_size': [200]
        }


chain=RegressorChain(SVR())

model=make_pipeline(StandardScaler(), chain)


grid=GridSearchCV(model,param_grid=params,cv=5,scoring=scorer)

grid.fit(X,y)
print(grid.score(X,y))
#grid.fit(X_result,y_result)

print(grid.best_params_)
#print(grid.score(X,y))
print(grid.score(X_result,y_result))

plt.figure()
#plt.plot(covid.index,covid['cas_confirmes'])
plt.plot(covid.index,covid['deces'])
plt.plot(covid.index,covid['reanimation'])
#plt.plot(covid.index,covid['hospitalises'])
plt.plot(X_result,grid.predict(X_result)[:,1:])
plt.show()
#######partial_fit
