#%%
from WaveSpace.WaveAnalysis import OpticalFlow
from WaveSpace.Utils import ImportHelpers
from WaveSpace.PlottingHelpers import Plotting

from __future__ import division
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


#%%
# Load data 
timeseries = ImportHelpers.load_wavedata_object("ExampleData/WaveData_SIM_mixedwaves_onset300_highSNR")
timeseries.set_active_dataBucket("SimulatedData")
applyGaussianBlur = False
OpticalFlow.create_uv(timeseries, applyGaussianBlur) 

UV = timeseries.DataBuckets["UV"].get_data()
SquareUV = np.reshape(UV, (UV.shape[0], int(np.sqrt(UV.shape[1])), int(np.sqrt(UV.shape[1])), UV.shape[2]))

OpticalFlow.calculate_directional_stability(timeseries)
DirectionalStabilitySeries = timeseries.DataBuckets["Directional_Stability_Timeseries"].get_data()
SquareDirectionalStabilitySeries = np.reshape(DirectionalStabilitySeries, (DirectionalStabilitySeries.shape[0], int(np.sqrt(DirectionalStabilitySeries.shape[1])), int(np.sqrt(DirectionalStabilitySeries.shape[1])), DirectionalStabilitySeries.shape[2]))
print(timeseries)

#%% plotting Angle

# which trial to visualise in this example
selectedTrial = 0
selectedTimePoint = 10

cmap = 'copper'
plt.figure(figsize=(10, 10))
plt.grid(None)
plt.imshow(np.angle(SquareUV[selectedTrial,:, :, selectedTimePoint]), cmap=cmap)
plt.title('HS angle')
plt.colorbar()

#%% plotting directional vectors
selectedTimePoint =200
SquareSimulatedData= np.reshape(timeseries.DataBuckets["SimulatedData"].get_data(), (timeseries.DataBuckets["SimulatedData"].get_data().shape[0], int(np.sqrt(timeseries.DataBuckets["SimulatedData"].get_data().shape[1])), int(np.sqrt(timeseries.DataBuckets["SimulatedData"].get_data().shape[1])), timeseries.DataBuckets["SimulatedData"].get_data().shape[2]))
cmap = 'copper'
plt.figure(figsize=(10, 10))
plt.grid(None)
plt.imshow(SquareSimulatedData[selectedTrial,:, :, selectedTimePoint], cmap=cmap)
plt.quiver(np.real(SquareUV[selectedTrial, :, :, selectedTimePoint]), np.imag( SquareUV[selectedTrial, :, :, selectedTimePoint]))

#%% make streamlines from pre-defined seedpoints
TrialNr = 2

# select a single timeseries
singleTrial = SquareUV[TrialNr,:,:,:] / np.abs(SquareUV[TrialNr,:,:,:])
WindowSize = 100

seedpoints = np.zeros(singleTrial.shape, dtype=bool)
seedpoints[:,:,:] = False
seedpoints[:,:,0] = True
out = Plotting.plot_streamlines(singleTrial, seedpoints)
out.show()

#%% Animate polar plot based on flattened UV to show direction consistency

selectedTrial = 0

def AnimatePolarScatter(frameNr, UV, AverageVectors , WindowSize):
    currentUV = UV[:, :, frameNr:frameNr + WindowSize]
    offsetArray = np.stack((np.angle(currentUV).ravel(), np.abs(currentUV).ravel()), axis=1)
    scatterPlot.set_offsets(offsetArray)
    scatterPlot.set_sizes(offsetArray[:, 1] * 30)
    if(frameNr >= WindowSize):
        currentAverages = AverageVectors[:,:,:,(frameNr-WindowSize)+1].ravel()
        for idx, line in enumerate(lines):
            line.set_data([0, np.angle(currentAverages[idx])],
                        [0, np.abs(currentAverages[idx])])

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
# ax.set_xlim(1.1)
ax.set_ylim(0, 1.1)
nTrials,dimx, dimy, nFrames = SquareUV.shape
# Note: This will effectively eliminate transients, because we are looking for a consistent relationship between pixels over time
WindowSize = 10
pad = np.zeros((nTrials, dimx, dimy, WindowSize))
normalizedUV = SquareUV / np.abs(SquareUV)
paddedUnitVec = np.concatenate((pad, normalizedUV, pad), axis=3)[0,:,:,:]
cmap = plt.cm.get_cmap('copper')
colorslice = cmap(np.linspace(0, 1, dimx * dimy))
allcolors = np.repeat(colorslice, WindowSize, axis=0)
alphasteps = np.linspace(0.1, 1, WindowSize)
alphasteps = np.repeat(alphasteps, dimx * dimy)
allcolors[:, 3] = alphasteps
scatterPlot = ax.scatter(np.angle(pad[selectedTrial,:,:,:]),
                         np.abs(pad[selectedTrial,:,:,:]), s=20, color=allcolors)
lines = ax.plot([np.zeros(paddedUnitVec.shape[0] * paddedUnitVec.shape[1]),
                 np.zeros(paddedUnitVec.shape[0] * paddedUnitVec.shape[1])],
                [np.zeros(paddedUnitVec.shape[0] * paddedUnitVec.shape[1]),
                 np.zeros(paddedUnitVec.shape[0] * paddedUnitVec.shape[1])],marker='o', linewidth=1.5, markersize=8)
for idx, line in enumerate(lines):
    line.set_color(colorslice[idx])

ani = animation.FuncAnimation(plt.gcf(),
                              AnimatePolarScatter, fargs=(paddedUnitVec, SquareDirectionalStabilitySeries, WindowSize),
                              frames=nFrames-1, interval=200)


#%%  Warning: Takes long
ani.save('ExampleData/Output/SimScatter.mp4')

#%% When you have a UV and raw data you can perform source sink analysis
poincareSinkSource = OpticalFlow.find_sources_sinks(timeseries, "SimulatedData")

# %%