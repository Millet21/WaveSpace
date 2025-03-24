# # Add the project root directory to the Python path when working with source code, 
# # not necessary when package is installed
# import sys
# import os
# path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, path )
# print(path)
#%%
from WaveSpace.Simulation import SimulationFuns
from WaveSpace.Utils import ImportHelpers

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# %% PLot an example wave, keep in mind that with long simulations and high sampling rates this could take long

timeseries = ImportHelpers.load_wavedata_object("ExampleData/WaveData_SIM_planewaves_onset300_lowSNR")
print(timeseries)

matrixSize = 16

trlToPlot = 2
def AnimatePlot(frameNR, fullstatus):
    img.set_data(fullstatus[:, :, frameNR])
    
    # lineseriesdata[:][frameNR] = fullstatus[probepositions[:, 0], probepositions[:, 1], frameNR]
    # lineseriesdata[:][frameNR] += np.arange(len(probepositions)) * linedistance
    for ind, position in enumerate(probepositions):
        lineseriesdata[ind][frameNR] = fullstatus[position[0],position[1],frameNR]
        lineseriesdata[ind][frameNR] += ind * linedistance
    currentPlot.cla()
    currentPlot.set_ylim(-1.2, len(probepositions * linedistance) +.2)
    #currentPlot.set_yticks(np.arange(0,len(probepositions)*linedistance,linedistance),["O" for probe in probepositions])
    #ax1.tick_params(axis='y', colors=['red', 'black'], )  
    currentPlot.yaxis.set_visible(False)
    currentPlot.plot(lineseriesdata.T, linewidth =4)
    for ind, line in enumerate(currentPlot.get_lines()):         
        line.set_color("black")
        #line.set_color(probecolors[ind])
        currentPlot.add_patch(plt.Rectangle((-2.5, (ind*linedistance)-0.25), 1, 0.5, facecolor='none',edgecolor=probecolors[ind],lw=8, clip_on=False))
    #currentPlot.get_lines()[3].set_color("red")

plotGridSize = (1,2)
plt.rcParams["figure.autolayout"] = True
fig = plt.figure(figsize=(plotGridSize[1]*8, plotGridSize[0]*8))
#IMSHOW grid
ax1 = plt.subplot2grid(plotGridSize, (0, 0), colspan=1, rowspan=1)
ax1.grid(None)
plt.set_cmap('copper')  
#plt.tight_layout()
ax1.axis('off')


#nframes = 30

probepositions = [(7,0),(7,2),(7,4),(7,6),(7,8),(7,10),(7,12),(7,14)]

# Mark Probes
lengthOfMatrix = matrixSize * matrixSize
# make all black
probecolors = []
allEdgeColors = [(0.0, 0.0, 0.0)for i in range(lengthOfMatrix)]
for ind, probe in enumerate(probepositions):
    currentColor = SimulationFuns.getProbeColor(ind, len(probepositions))
    currentRect = plt.Rectangle((probe[1]-0.5, probe[0]-0.5), 1, 1, facecolor='none',edgecolor=currentColor,lw=2)
    probecolors.append(currentRect.get_edgecolor())
    ax1.add_patch(currentRect)

currentShape = data = timeseries.get_active_data().shape
timeseries.DataBuckets["SimulatedData"].reshape((currentShape[0],matrixSize,matrixSize,currentShape[2]), "trl_posx_posy_time")
data = timeseries.get_active_data()
nframes = data.shape[3]
lineseriesdata = np.zeros((len(probepositions), nframes), dtype='float64')
currentPlot = plt.subplot2grid(
        plotGridSize, (0,1), colspan=1, rowspan=1)
currentPlot.plot(range(nframes), lineseriesdata.T,linewidth=3)
currentPlot.grid(visible = False)
currentPlot.set_ylabel([])
currentPlot.set_facecolor("white")
linedistance = 2

fullstatus = data[trlToPlot] # Take single trial
img =  ax1.imshow(fullstatus[:, :, 0], origin='lower',vmin=-1, vmax=1)
ani = animation.FuncAnimation(plt.gcf(),
                              AnimatePlot, fargs=(fullstatus,),
                              frames=nframes, interval=50)

plt.plot(data[0,8,8,:])
#plt.show()
# %%
ani.save('ExampleData/Output/ExampleTargetWave.mp4')

# %%