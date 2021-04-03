import pandas as pd

### décale un matrice de données indéxées par un date à n jours en avance
def timeshift_day(data,n):
    data.index = data.index.shift(periods = n, freq ='D')