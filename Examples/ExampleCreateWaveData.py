# # Add the project root directory to the Python path when working with source code, 
# # not necessary when package is installed
# import sys
# import os
# path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, path )
# print(path)
#%%
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from WaveSpace.Utils import WaveData as wd
import scipy.io as sp
import numpy as np

#%%
# [ToDo] Eventually we're going to make up an np-array or something
sampleRate = 1000

data = sp.loadmat('ExampleData/timeseries.mat')
data = data['Data']
data = data.reshape(3,256,500)
time = np.linspace(0,(data.shape[-1]/sampleRate)-(1/sampleRate),data.shape[-1])

waveData = wd.WaveData(time=time)
waveData.set_channel_names([str(i) for i in np.arange(data.shape[1])])
dataBucket = wd.DataBucket(data, "RawTimeSeries","chan_time", waveData.get_channel_names())
waveData.add_data_bucket(dataBucket)

waveData.set_sample_rate(sampleRate)

waveData.save_to_file("ExampleData/Output/TestData")
# print __repr__
print(waveData)
# %%