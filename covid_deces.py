##Methode SVR##
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split  #sklearn c'est pour le machine learning et la on prends de quoi séparer nos données
from sklearn.pipeline import make_pipeline  #permet de réunir la normalisation et l'algorithme en une seule fonction
from sklearn.preprocessing import StandardScaler  #permet de normaliser les données pour que ça fonctionne mieu
from sklearn.linear_model import ElasticNet #l'algorithme qu'on utilise
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from dist_moy import dist_moy

##############################################################################
svr = SVR()
params = {'svr__C' : [10**i for i in range(6,7)]}#,
          #'svr__epsilon' : [0.003],
          #'svr__gamma' : [i for i in [1000, 100, 10, 0.1, 0.001, 'scale']]}
##########################Traitement de données################################
cdata = pd.read_csv('covid_numbers.csv',index_col='date',parse_dates=True)
cdata = cdata[cdata['granularite']=='pays']

for i in cdata:
    if not (cdata[i].name in ["deces"]):#,"cas_confirmes"]):
        cdata.drop([i], axis=1, inplace = True)

cdata = cdata.dropna(axis=0)

print(cdata.count())
cdata

#########################Rangement de données##################################
date = '2020-10-15'

y = cdata[:date]
X = pd.to_datetime(y.index)

size = len(X)

y = y.values.reshape(size,)
X = X.values.reshape(size,1)

X_train, y_train = X, y

model = make_pipeline(StandardScaler(), svr)  #ça sera ça notre algorithme
scorer = make_scorer(mean_squared_error, greater_is_better=False)
tscv = TimeSeriesSplit(n_splits=10) #Decoupage pour le CV adapté aux series temporelles.
grid = GridSearchCV(model, params, scorer, cv = tscv )

grid.fit(X_train, y_train)

x_prevu = pd.to_datetime(cdata[date:].index)
x_prevu = x_prevu.values.reshape(len(x_prevu),1)

cdata['deces'].plot(label="deces reels")
#cdata['cas_confirmes'].plot(label="cas reels")
plt.plot(x_prevu,grid.predict(x_prevu),label="valeur predites")
plt.plot(X_train, grid.predict(X_train), label = "valeur predites")
plt.legend()
plt.show()
print(grid.best_params_)

######LE CODE COMPILE#######