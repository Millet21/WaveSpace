#%%
from WaveSpace.Decomposition import EMD
from WaveSpace.Utils import ImportHelpers, HelperFuns as hf,WaveData as wd
from WaveSpace.Simulation import SimulationFuns
from WaveSpace.PlottingHelpers import Plotting as plotting

import numpy as np

#%% Single Unidirectional waves and some noise, high SNR
Type =  "PlaneWave" # PlaneWave	 StationaryPulse   TargetWave	RotatingWave	LocalOscillation	SpatialPinkNoise	WhiteNoise
nTrials = 3
matrixSize = 16
SampleRate= 500
SimDuration= 1.0

SpatialFrequency = [0.25, 0.5, 0.75]
TemporalFrequency = [10, 10, 10] 
WaveDirection = [45,45,45]  #lower or higher than 0 determines in or outward motion for targetWave and ClockWise or Counter ClockWise for RotatingWave 
#initialize data
onsetInMs = 300
WaveOnset =  [(onsetInMs / (1000/SampleRate) ) / (1 / tf  * SampleRate)for tf in TemporalFrequency]
WaveDuration = 10
wave = SimulationFuns.simulate_signal(
    Type, 
    nTrials, 
    matrixSize, 
    SampleRate, 
    SimDuration, 
    #SimOptions from here on
    TemporalFrequency = TemporalFrequency,
    SpatialFrequency = SpatialFrequency,
    WaveDirection = WaveDirection,
    WaveOnset = WaveOnset,
    WaveDuration = WaveDuration
    )

noise = SimulationFuns.simulate_signal("SpatialPinkNoise", nTrials, matrixSize, SampleRate, SimDuration)

SNR = 2.0
SimOutput = SimulationFuns.SNRMix(wave, noise, SNR)
SimOutput._simInfo.append({"SNRMix" : [{"NoiseType": "SpatialPinkNoise"}, {"SNR": SNR}, {"matrixSize": matrixSize}, {"sampleRate": SampleRate}]})
filename ="ExampleData/WaveData_SIM_test"
SimOutput.save_to_file(filename)


#%%
waveData = ImportHelpers.load_wavedata_object("ExampleData/WaveData_SIM_test")
print(waveData)

EMD.EMD(waveData, nIMFs = 7, dataBucketName="SimulatedData")
plotting.plot_imfs(waveData.get_data("AnalyticSignal")[0,1].T, 4, waveData.get_time())

IF, IA, IP = EMD.FreqAmpPhaseFromAnalytic(waveData, 5, 3)
FOI = 10
FOIind, Mean_freq=EMD.find_nearest_to_FOI(waveData, IF, FOI, start_time=0.5, end_time=1)

_, num_trials, num_channels,  nTime = waveData.get_data('AnalyticSignal').shape
SubsetData = np.empty((num_trials, num_channels, nTime), dtype=complex)  
for trial in range(num_trials):
    for chan in range(num_channels):
        SubsetData[trial, chan, :] = waveData.get_data('AnalyticSignal')[FOIind[trial, chan], trial, chan, :]
#make a temporary DataBucket 
SubsetBucket = wd.DataBucket(SubsetData, "AnalyticSignalSubset", "trl_chan_time" ,  
                                waveData.DataBuckets[waveData.ActiveDataBucket].get_channel_names())
waveData.add_data_bucket(SubsetBucket)
waveData.log_history(["Subset","Original = AnalyticSignal"])

#%%
waveData.DataBuckets["AnalyticSignalSubset"].set_data(waveData.get_data("AnalyticSignalSubset")[:,:,200:499], "trl_chan_time")
hf.relative_phase(waveData,  ref=1, dataBucketName = "AnalyticSignalSubset")