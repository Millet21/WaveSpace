#%%
from WaveSpace.Utils import ImportHelpers
from WaveSpace.PlottingHelpers import Plotting
from WaveSpace.WaveAnalysis import WaveAnalysis as wa

import numpy as np
import matplotlib.pyplot as plt

#%%
savefolder = "ExampleData/Output/"

TimeSeries= ImportHelpers.load_wavedata_object("ExampleData/WaveData_SIM_planewaves_onset300_lowSNR")

TimeSeries.DataBuckets["SimulatedData"].reshape((3,16,16,TimeSeries.DataBuckets["SimulatedData"].get_data().shape[2]), "trl_posx_posy_time")


#channelIndeces = [(0,4),(1,4),(2,4),(3,4),(4,4),(5,4),(6,4)]
channelIndeces = [(7,0),(7,1),(7,2),(7,4),(7,6),(7,8),(7,10),(7,12),(7,13),(7,14)]

#restrict to (temp) frequencies between lower and upper bound:
lowerBound = 2
upperBound = 40

wa.FFT_2D(TimeSeries, channelIndeces, lowerBound, upperBound)

result = TimeSeries.get_data("Result")

# Get the number of trials
n_trials = TimeSeries.get_data("SimulatedData").shape[0]

for trial in range(n_trials):

    logRatios=np.log(result["Max Along Power"][trial]/result["Max Reverse Power"][trial])
    newlineseries = np.zeros((len(channelIndeces),TimeSeries.get_data("SimulatedData").shape[3]))
    for ind, position in enumerate(channelIndeces):
        newlineseries[ind] = TimeSeries.get_data("SimulatedData")[trial, position[0], position[1], :]
    plt.figure()
    plt.imshow(newlineseries, aspect=4)
    plt.title(f"Channels over time (Trial {trial+1})")
    #plt.savefig(savefolder+"TargetTest" +str(trial)+ ".png", dpi=300)

    plt.show()

    plot = Plotting.plotfft_zoomed(TimeSeries.get_data("FFT_ABS")[trial,:,:], TimeSeries.get_sample_rate(), -20, 20, "fft abs", scale='log')
    #plot.savefig(savefolder+"TargetTest" +str(trial)+ "FFT.png", dpi=300)

    plot.show()

    x_labels = np.arange(1)
    plt.figure()
    plt.bar(x_labels, [result["Max Along Power"][trial]], color='b', width = 0.25 )
    plt.bar(x_labels + 0.25, [result["Max Reverse Power"][trial]] , color='r', width = 0.25 )
    plt.legend(labels=["Along", "Reverse"])
    plt.xticks(x_labels + 0.125, ["0 degree"])
    plt.title(f"Max PoweWr (Trial {trial+1})")
    plt.show()


# %%
sfreq = TimeSeries .get_sample_rate()
trials, nChan,  nTimepoints = TimeSeries.get_data("FFT_ABS").shape
spatialFreqAxis = nChan/2 * np.linspace(-1, 1, nChan)
tempFreqAxis = np.arange(-sfreq/2, sfreq/2, 1/(nTimepoints/sfreq))
min_val = -20
max_val = 20
plotrange = np.where((tempFreqAxis > min_val) & (tempFreqAxis < max_val))
standAverageFFT = np.average(TimeSeries.get_data("FFT_ABS"), axis=0)

plt.grid(True)
plt.imshow(newlineseries,aspect=22)
plt.title("Channels over time")
plt.colorbar()
#plt.savefig(savefolder+"FrontBack_channels_time_standing.png", dpi=300)
plt.show()
plt.figure(figsize=(8,2))
plot = Plotting.plotfft_zoomed(standAverageFFT, TimeSeries.get_sample_rate(), -20, 20, "Standing Average",scale='log')
plt.grid(True)

#plot.savefig(savefolder+"FrontBack_standAverageFFT.png", dpi=300)
plot.show()
plot.close()

x_labels = np.arange(1)
plt.figure()
plt.bar(x_labels, [result["Max Along Power"][trial]], color='b', width = 0.25 )
plt.bar(x_labels + 0.25, [result["Max Reverse Power"][trial]] , color='r', width = 0.25 )
plt.legend(labels=["Along", "Reverse"])
plt.xticks(x_labels + 0.125, ["0 degree"])
plt.title(f"Max Power (Trial {trial+1})")
plt.show()
