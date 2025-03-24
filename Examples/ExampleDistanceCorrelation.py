#%%
from WaveSpace.WaveAnalysis import DistanceCorrelation
from WaveSpace.Utils import ImportHelpers, HelperFuns
from WaveSpace.Decomposition import GenPhase 
from WaveSpace.SpatialArrangement import SensorLayout 

import numpy as np
import matplotlib.pyplot as plt
#%%
timeSeries= ImportHelpers.load_wavedata_object("ExampleData\WaveData_SIM_planewave_plus_localOscillators_highSNR")
dataShape = timeSeries.DataBuckets["SimulatedData"].get_data().shape 
order = np.random.choice(dataShape[1],dataShape[1],replace=False)
shape = ((int(np.sqrt(len(order))),int(np.sqrt(len(order)))))
HelperFuns.order_to_grid(timeSeries, shape, timeSeries.DataBuckets["SimulatedData"].get_dimord())
GenPhase.generalized_phase(timeSeries)
print(timeSeries)
#%%
SensorLayout.regularGrid(timeSeries, timeSeries.get_channel_positions())
SensorLayout.distmat_to_2d_coordinates_Isomap(timeSeries)
DistanceCorrelation.calculate_distance_correlation(timeSeries)
Result = timeSeries.DataBuckets["PhaseDistanceCorrelation"].get_data()
#Get the timepoint where the correlation is maximal and the corresponding sourcepoints (in space)
filteredResult = Result[(Result['trialind']==0)]
x, y, evaluationpoint = filteredResult[(filteredResult['rho']*(1-filteredResult['p'])==(filteredResult['rho']*(1-filteredResult['p'])).max())][['sourcepointsX','sourcepointsY','evaluationpoints']].values[0]

#Plot a wave-example 
fig, ax = plt.subplots()
data = timeSeries.DataBuckets["ComplexPhaseData"].get_data()[0, :, :, int(evaluationpoint)]
im = ax.imshow(np.angle(data), cmap='twilight', vmin=-np.pi, vmax=np.pi, extent=[0, 16, 0, 16])
cbar = plt.colorbar(im, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cbar.ax.set_yticklabels(['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Phasemap at evaluation timepoint: {}'.format(evaluationpoint))
ax.scatter(x, y, s=40, facecolors='none', edgecolors='r', linewidths=3)
plt.show()
# %%