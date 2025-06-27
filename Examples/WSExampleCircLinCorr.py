"""
Circular-linear correlation. Uses a Python implementation based on https://github.com/mullerlab/generalized-phase.git
Requires complex data in a regular grid with known (relative) distances between grid points. 

"""
#%%
import sys
import os

# Add the project root directory to the Python path when working with source code, 
# not necessary when package is installed
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path )
print(path)

from WaveSpace.PlottingHelpers import Plotting
from WaveSpace.Utils import HelperFuns as hf
from WaveSpace.Utils import ImportHelpers
from WaveSpace.WaveAnalysis import DistanceCorrelation
from WaveSpace.SpatialArrangement import SensorLayout as sensors

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib import colormaps
# Load some simulated data
dataPath  = os.path.join(path, "Examples/ExampleData/Output") 
waveData = ImportHelpers.load_wavedata_object(dataPath + "/ComplexData")

#%% 
# we already know that our data is on a regular grid because we generated it that way
# so we can simply use the channel positions to create a distance matrix

sensors.regularGrid(waveData)

#%%
DistanceCorrelation.calculate_distance_correlation_GP(waveData, dataBucketName = "AnalyticSignal", evaluationAngle=np.pi, tolerance=0.2)

#%% 
pointRange = range(0,20,2)
sourcePoints = []
for i in pointRange:
    sourcePoints.append([i,i])
#find potential wave starting points 
DistanceCorrelation.calculate_distance_correlation(waveData, dataBucketName = "AnalyticSignal", sourcePoints=sourcePoints, pixelSpacing=1)

phaseDistCorr= waveData.get_data("PhaseDistanceCorrelation")

selectedTrial = 4
for point in sourcePoints:
    rho = phaseDistCorr.loc[(phaseDistCorr["trialInd"] == selectedTrial) & (phaseDistCorr["sourcePointX"] == point[0]) & (phaseDistCorr["sourcePointY"] == point[1])]
    plt.plot(rho["rho"].tolist())
plt.legend([str(sourcePoint) for sourcePoint in sourcePoints])
plt.show()
