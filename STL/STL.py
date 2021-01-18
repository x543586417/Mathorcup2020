import statsmodels.api as sm
import matplotlib.pyplot as plt
import time
import pandas as pd

dta = pd.read_csv(r"/STL/STLdata1.csv", usecols=['Hour', 'Upload'])
dta = dta.set_index('Hour')
dta['Upload'] = dta['Upload'].apply(pd.to_numeric, errors='ignore')
dta.Upload.interpolate(inplace=True)
res = sm.tsa.seasonal_decompose(dta.Upload, freq=288)
res.plot()
plt.show()
