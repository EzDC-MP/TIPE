''''              Predictions a l'aide du module Prophet                    '''

import matplotlib.pyplot as plt
import pandas as pd
import re
from fbprophet import Prophet
## 	debug ##
def clean_dates(D):
    V = D.ds.values
    txt_d = "[0123456789]{4}-[0123456789]{2}-[0123456789]{2}"
    for i in range(len(V)):
        x = re.search(txt_d, V[i])
        if x == None:
            cdata.drop(index=i, inplace = True)
    return
## Importation des donnees ##
'''
On s'interessera que sur les donnees sur les deces pour l'instant
'''

cdata = pd.read_csv('chiffres-cles.csv',index_col='date',parse_dates=True)
cdata = cdata[cdata['granularite']=='pays']

for i in cdata:
    if not (cdata[i].name in ["deces"]):#,"cas_confirmes"]):
        cdata.drop([i], axis=1, inplace = True)

cdata = cdata.dropna(axis=0)
cdata.reset_index(level=0, inplace=True)
cdata.columns = ['ds', 'y']
clean_dates(cdata)

### Importation du Modele ##
model = Prophet()

## Prediction ##
date = '2020-12-06' ##max 2020-12-29
d = 23 #nombre de jour entre la date donnée et la date max de cdata
cdata_less = cdata[cdata['ds'] <= date]
model.fit(cdata_less)


future_df = model.make_future_dataframe(periods=d)
prediction = model.predict(future_df)

## Plot ##
pred = prediction[['ds','yhat']]

cdata = cdata.set_index('ds')
cdata = cdata.groupby(['ds']).max()
pred = pred[pred['ds'] > date]
pred = pred.set_index('ds')

date_x = cdata.index
plt.plot(date_x, cdata['y'],label="deces reels")
plt.plot([i for i in date_x if i > date], pred['yhat'], label = "valeures predites")
plt.legend()
plt.show()