import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso


covid = pd.read_excel('covid1.xlsx',index_col='date',parse_dates=True)

covid = covid[covid['granularite']=='pays']

covid.drop(['cas_ehpad', 'cas_confirmes_ehpad', 'cas_possibles_ehpad','source_url',
            'deces_ehpad', 'reanimation',  'gueris','source_nom',
            'nouvelles_hospitalisations', 'nouvelles_reanimations','depistes',
             'maille_code', 'maille_nom','source_archive','source_type'],inplace=True,axis=1)

covid = covid.dropna(axis=0)

subject=['deces']

jour_prevu=75
size=covid.count()[0]-jour_prevu
X = covid.index[:size]
y = covid[:size][subject]


y = y.values.reshape(size)
X = X.values.reshape(size,1)

X_train, y_train=X, y

model=make_pipeline(StandardScaler(), SVR())
params={
    'svr__degree': [0],
    'svr__C': [600000],
}

grid = GridSearchCV(model,param_grid=params,cv=4)

grid.fit(X_train , y_train)


x_prevu = covid.index[size:]
x_prevu = x_prevu.values.reshape(len(x_prevu),1)







