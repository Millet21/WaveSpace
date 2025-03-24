#%%
import sys
import os

# Add the project root directory to the Python path when working with source code, 
# not necessary when package is installed
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path )
print(path)

from WaveSpace.Simulation import SimulationFuns
from WaveSpace.PlottingHelpers import Plotting
from WaveSpace.Utils import HelperFuns as hf

import numpy as np
import matplotlib.pyplot as plt

#%% Single Unidirectional waves and some noise, high SNR
Type =  "PlaneWave" # PlaneWave	 StationaryPulse   TargetWave	RotatingWave	LocalOscillation	SpatialPinkNoise	WhiteNoise
nTrials = 3
matrixSize = 20
SampleRate= 500
SimDuration= 1.0

SpatialFrequency = [0.25, 0.5, 0.75]
TemporalFrequency = [10, 20, 17.5] 
WaveDirection = [90,0,-30]  #lower or higher than 0 determines in or outward motion for targetWave and ClockWise or Counter ClockWise for RotatingWave 
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
filename ="./ExampleData/WaveData_SIM_planewaves_onset300_highSNR"
SimOutput.save_to_file(filename)
#save(SimOutput.get_data("SimulatedData"), filename + "_data")
#save(SimOutput.get_SimInfo(),filename + "_simInfo")

data = SimOutput.get_data("SimulatedData")
data = np.reshape(data ,(nTrials,matrixSize,matrixSize,int(np.floor(SampleRate*SimDuration))))
trialNr =2
fullstatus = data[trialNr] # Take single trial
print(SimOutput)
plt.imshow(fullstatus[:, :, 0], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.imshow(fullstatus[:, :, 200], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.plot(fullstatus[8,8,:])
plt.show()


#%% Single Unidirectional waves and some noise, low SNR
Type =  "PlaneWave" # PlaneWave	 StationaryPulse   TargetWave	RotatingWave	LocalOscillation	SpatialPinkNoise	WhiteNoise
nTrials = 3
matrixSize = 16
SampleRate= 500
SimDuration= 1.0

SpatialFrequency = [0.25, 0.5, 0.75]
TemporalFrequency = [10, 20, 17.5] 
WaveDirection = [30,0,-30]  #lower or higher than 0 determines in or outward motion for targetWave and ClockWise or Counter ClockWise for RotatingWave 
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

SNR = 0.6
SimOutput = SimulationFuns.SNRMix(wave, noise, SNR)
SimOutput._simInfo.append({"SNRMix" : [{"NoiseType": "SpatialPinkNoise"}, {"SNR": SNR}, {"matrixSize": matrixSize}, {"sampleRate": SampleRate}]})
SimOutput.save_to_file("./ExampleData/WaveData_SIM_planewaves_onset300_lowSNR")
filename ="./ExampleData/WaveData_SIM_planewaves_onset300_lowSNR"
SimOutput.save_to_file(filename)
# save(SimOutput.get_data("SimulatedData"), filename + "_data")
# save(SimOutput.get_SimInfo(),filename + "_simInfo")
data = SimOutput.get_data("SimulatedData")
data = np.reshape(data ,(nTrials,matrixSize,matrixSize,int(np.floor(SampleRate*SimDuration))))
trialNr =2
fullstatus = data[trialNr] # Take single trial
print(SimOutput)
plt.imshow(fullstatus[:, :, 0], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.imshow(fullstatus[:, :, 200], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.plot(fullstatus[8,8,:])
plt.show()


#%% Multiple Unidirectional waves and some noise, high SNR
Type =  "PlaneWave" # PlaneWave	 StationaryPulse   TargetWave	RotatingWave	LocalOscillation	SpatialPinkNoise	WhiteNoise
nTrials = 3
matrixSize = 16
SampleRate= 500
SimDuration= 1.0

SpatialFrequency = [0.25, 0.5, 0.75]
TemporalFrequency = [10, 20, 17.5] 
WaveDirection = [30,0,-30]  #lower or higher than 0 determines in or outward motion for targetWave and ClockWise or Counter ClockWise for RotatingWave 
#initialize data
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
    #CenterX = CenterX,
    #CenterY = CenterY, 
    #Sigma=Sigma
    WaveOnset = WaveOnset,
    WaveDuration = WaveDuration
    )

SpatialFrequency = [0.25, 0.5, 0.75]
TemporalFrequency = [20, 17.5, 10] 
WaveDirection = [45,90,45]

wave2 = SimulationFuns.simulate_signal(
    Type, 
    nTrials, 
    matrixSize, 
    SampleRate, 
    SimDuration, 
    #SimOptions from here on
    TemporalFrequency = TemporalFrequency,
    SpatialFrequency = SpatialFrequency,
    WaveDirection = WaveDirection,
    #CenterX = CenterX,
    #CenterY = CenterY, 
    #Sigma=Sigma
    WaveOnset = WaveOnset,
    WaveDuration = WaveDuration
    )


SNR = 2

alldata = (wave.DataBuckets["SimulatedData"].get_data() + wave2.DataBuckets["SimulatedData"].get_data()) / 2
wave.DataBuckets["SimulatedData"].set_data(alldata, "trl_chan_time")

allmasks = (wave.DataBuckets["Mask"].get_data() + wave2.DataBuckets["Mask"].get_data()) / 2
wave.DataBuckets["Mask"].set_data(allmasks, "trl_chan_time")
wave._simInfo.append({"Secondary waves" : wave2.get_SimInfo()})

noise = SimulationFuns.simulate_signal("SpatialPinkNoise", nTrials, matrixSize, SampleRate, SimDuration)
SimOutput = SimulationFuns.SNRMix(wave,noise, SNR)
SimOutput._simInfo.append({"SNRMix" : [{"NoiseType": "SpatialPinkNoise"}, {"SNR": SNR}, {"matrixSize": matrixSize}, {"sampleRate": SampleRate}]})
SimOutput.save_to_file("./ExampleData/WaveData_SIM_mixedwaves_onset300_highSNR")
filename ="./ExampleData/WaveData_SIM_mixedwaves_onset300_highSNR"
SimOutput.save_to_file(filename)
# save(SimOutput.get_data("SimulatedData"), filename + "_data")
# save(SimOutput.get_SimInfo(),filename + "_simInfo")

data = SimOutput.get_data("SimulatedData")
data = np.reshape(data ,(nTrials,matrixSize,matrixSize,int(np.floor(SampleRate*SimDuration))))
trialNr =2
fullstatus = data[trialNr] # Take single trial
print(SimOutput)
plt.imshow(fullstatus[:, :, 0], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.imshow(fullstatus[:, :, 200], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.plot(fullstatus[8,8,:])
plt.show()

#%% Multiple Unidirectional waves and some noise, low SNR
Type =  "PlaneWave" # PlaneWave	 StationaryPulse   TargetWave	RotatingWave	LocalOscillation	SpatialPinkNoise	WhiteNoise
nTrials = 3
matrixSize = 16
SampleRate= 500
SimDuration= 1.0

SpatialFrequency = [0.25, 0.5, 0.75]
TemporalFrequency = [10, 20, 17.5] 
WaveDirection = [30,0,-30]  #lower or higher than 0 determines in or outward motion for targetWave and ClockWise or Counter ClockWise for RotatingWave 
#initialize data
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
    #CenterX = CenterX,
    #CenterY = CenterY, 
    #Sigma=Sigma
    WaveOnset = WaveOnset,
    WaveDuration = WaveDuration
    )

SpatialFrequency = [0.25, 0.5, 0.75]
TemporalFrequency = [20, 17.5, 10] 
WaveDirection = [45,90,45]

wave2 = SimulationFuns.simulate_signal(
    Type, 
    nTrials, 
    matrixSize, 
    SampleRate, 
    SimDuration, 
    #SimOptions from here on
    TemporalFrequency = TemporalFrequency,
    SpatialFrequency = SpatialFrequency,
    WaveDirection = WaveDirection,
    #CenterX = CenterX,
    #CenterY = CenterY, 
    #Sigma=Sigma
    WaveOnset = WaveOnset,
    WaveDuration = WaveDuration
    )


SNR = 0.6

alldata = (wave.DataBuckets["SimulatedData"].get_data() + wave2.DataBuckets["SimulatedData"].get_data()) / 2
wave.DataBuckets["SimulatedData"].set_data(alldata, "trl_chan_time")

allmasks = (wave.DataBuckets["Mask"].get_data() + wave2.DataBuckets["Mask"].get_data()) / 2
wave.DataBuckets["Mask"].set_data(allmasks, "trl_chan_time")
wave._simInfo.append({"Secondary waves" : wave2.get_SimInfo()})

noise = SimulationFuns.simulate_signal("SpatialPinkNoise", nTrials, matrixSize, SampleRate, SimDuration)
SimOutput = SimulationFuns.SNRMix(wave,noise, SNR)
SimOutput._simInfo.append({"SNRMix" : [{"NoiseType": "SpatialPinkNoise"}, {"SNR": SNR}, {"matrixSize": matrixSize}, {"sampleRate": SampleRate}]})
SimOutput.save_to_file("./ExampleData/WaveData_SIM_mixedwaves_onset300_lowSNR")
filename ="./ExampleData/WaveData_SIM_mixedwaves_onset300_lowSNR"
SimOutput.save_to_file(filename)
# save(SimOutput.get_data("SimulatedData"), filename + "_data")
# save(SimOutput.get_SimInfo(),filename + "_simInfo")

data = SimOutput.get_data("SimulatedData")
data = np.reshape(data ,(nTrials,matrixSize,matrixSize,int(np.floor(SampleRate*SimDuration))))
trialNr =2
fullstatus = data[trialNr] # Take single trial
print(SimOutput)
plt.imshow(fullstatus[:, :, 0], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.imshow(fullstatus[:, :, 200], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.plot(fullstatus[8,8,:])
plt.show()
# %% Planewave mixed with local oscillators high SNR

Type =  "PlaneWave" # PlaneWave	 StationaryPulse   TargetWave	RotatingWave	LocalOscillation	SpatialPinkNoise	WhiteNoise
nTrials = 3
matrixSize = 16
SampleRate= 500
SimDuration= 1.0

SpatialFrequency = [0.25, 0.5, 0.75]
TemporalFrequency = [10, 20, 17.5] 
WaveDirection = [30,0,-30]  #lower or higher than 0 determines in or outward motion for targetWave and ClockWise or Counter ClockWise for RotatingWave 
#initialize data
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


Type =  "LocalOscillation" # PlaneWave	 StationaryPulse   TargetWave	RotatingWave	LocalOscillation	SpatialPinkNoise	WhiteNoise
OscillatoryPhase = "Random"
TemporalFrequency = [20, 17.5, 10] 
WaveDirection = [45,90,45]

wave2 = SimulationFuns.simulate_signal(
    Type, 
    nTrials, 
    matrixSize, 
    SampleRate, 
    SimDuration, 
    #SimOptions from here on
    TemporalFrequency = TemporalFrequency,
    OscillatoryPhase = OscillatoryPhase
    )

alldata = (wave.DataBuckets["SimulatedData"].get_data() + wave2.DataBuckets["SimulatedData"].get_data()) / 2
wave.DataBuckets["SimulatedData"].set_data(alldata, "trl_chan_time")

wave._simInfo.append({"Local oscillators" : wave2.get_SimInfo()})
SNR = 2.0
noise = SimulationFuns.simulate_signal("SpatialPinkNoise", nTrials, matrixSize, SampleRate, SimDuration)
SimOutput = SimulationFuns.SNRMix(wave,noise, SNR)
SimOutput._simInfo.append({"SNRMix" : [{"NoiseType": "SpatialPinkNoise"}, {"SNR": SNR}, {"matrixSize": matrixSize}, {"sampleRate": SampleRate}]})
SimOutput.save_to_file("./ExampleData/WaveData_SIM_planewave_plus_localOscillators_highSNR")
filename ="./ExampleData/WaveData_SIM_planewave_plus_localOscillators_highSNR"
SimOutput.save_to_file(filename)
# save(SimOutput.get_data("SimulatedData"), filename + "_data")
# save(SimOutput.get_SimInfo(),filename + "_simInfo")

data = SimOutput.get_data("SimulatedData")
data = np.reshape(data ,(nTrials,matrixSize,matrixSize,int(np.floor(SampleRate*SimDuration))))
trialNr =2
fullstatus = data[trialNr] # Take single trial
print(SimOutput)
plt.imshow(fullstatus[:, :, 0], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.imshow(fullstatus[:, :, 200], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.plot(fullstatus[8,8,:])
plt.show()
# %%# %% Planewave mixed with local oscillators low SNR

Type =  "PlaneWave" # PlaneWave	 StationaryPulse   TargetWave	RotatingWave	LocalOscillation	SpatialPinkNoise	WhiteNoise
nTrials = 3
matrixSize = 16
SampleRate= 500
SimDuration= 1.0

SpatialFrequency = [0.25, 0.5, 0.75]
TemporalFrequency = [10, 20, 17.5] 
WaveDirection = [30,0,-30]  #lower or higher than 0 determines in or outward motion for targetWave and ClockWise or Counter ClockWise for RotatingWave 
#initialize data
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


Type =  "LocalOscillation" # PlaneWave	 StationaryPulse   TargetWave	RotatingWave	LocalOscillation	SpatialPinkNoise	WhiteNoise
OscillatoryPhase = "Random"
TemporalFrequency = [20, 17.5, 10] 
WaveDirection = [45,90,45]

wave2 = SimulationFuns.simulate_signal(
    Type, 
    nTrials, 
    matrixSize, 
    SampleRate, 
    SimDuration, 
    #SimOptions from here on
    TemporalFrequency = TemporalFrequency,
    OscillatoryPhase = OscillatoryPhase
    )

alldata = (wave.DataBuckets["SimulatedData"].get_data() + wave2.DataBuckets["SimulatedData"].get_data()) / 2
wave.DataBuckets["SimulatedData"].set_data(alldata, "trl_chan_time")

wave._simInfo.append({"Local oscillators" : wave2.get_SimInfo()})
SNR = 0.6
noise = SimulationFuns.simulate_signal("SpatialPinkNoise", nTrials, matrixSize, SampleRate, SimDuration)
SimOutput = SimulationFuns.SNRMix(wave,noise, SNR)
SimOutput._simInfo.append({"SNRMix" : [{"NoiseType": "SpatialPinkNoise"}, {"SNR": SNR}, {"matrixSize": matrixSize}, {"sampleRate": SampleRate}]})
SimOutput.save_to_file("./ExampleData/WaveData_SIM_planewave_plus_localOscillators_lowSNR")
filename ="./ExampleData/WaveData_SIM_planewave_plus_localOscillators_lowSNR"
SimOutput.save_to_file(filename)
# save(SimOutput.get_data("SimulatedData"), filename + "_data")
# save(SimOutput.get_SimInfo(),filename + "_simInfo")

data = SimOutput.get_data("SimulatedData")
data = np.reshape(data ,(nTrials,matrixSize,matrixSize,int(np.floor(SampleRate*SimDuration))))
trialNr =2
fullstatus = data[trialNr] # Take single trial
print(SimOutput)
plt.imshow(fullstatus[:, :, 0], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.imshow(fullstatus[:, :, 200], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.plot(fullstatus[8,8,:])
plt.show()
# %% Target wave moving out

Type =  "TargetWave" # PlaneWave	 StationaryPulse   TargetWave	RotatingWave	LocalOscillation	SpatialPinkNoise	WhiteNoise
nTrials = 3
matrixSize = 16
SampleRate= 500
SimDuration= 1.0

CenterX = 4
CenterY = 4

SpatialFrequency = [0.25, 0.5, 0.75]
TemporalFrequency = [10, 20, 17.5] 
WaveDirection = 1  #lower or higher than 0 determines in or outward motion for targetWave and ClockWise or Counter ClockWise for RotatingWave 
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
    WaveDuration = WaveDuration, 
    CenterX = CenterX,
    CenterY = CenterY
    )

noise = SimulationFuns.simulate_signal("SpatialPinkNoise", nTrials, matrixSize, SampleRate, SimDuration)

SNR = 2.0
SimOutput = SimulationFuns.SNRMix(wave, noise, SNR)
SimOutput._simInfo.append({"SNRMix" : [{"NoiseType": "SpatialPinkNoise"}, {"SNR": SNR}, {"matrixSize": matrixSize}, {"sampleRate": SampleRate}]})
SimOutput.save_to_file("./ExampleData/WaveData_SIM_targetwaves_highSNR")
filename ="./ExampleData/WaveData_SIM_targetwaves_highSNR"
SimOutput.save_to_file(filename)
# save(SimOutput.get_data("SimulatedData"), filename + "_data")
# save(SimOutput.get_SimInfo(),filename + "_simInfo")

data = SimOutput.get_data("SimulatedData")
data = np.reshape(data ,(nTrials,matrixSize,matrixSize,int(np.floor(SampleRate*SimDuration))))
trialNr =2
fullstatus = data[trialNr] # Take single trial
print(SimOutput)
plt.imshow(fullstatus[:, :, 0], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.imshow(fullstatus[:, :, 200], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.plot(fullstatus[8,8,:])
plt.show()

# %% TargetWave low snr

Type =  "TargetWave" # PlaneWave	 StationaryPulse   TargetWave	RotatingWave	LocalOscillation	SpatialPinkNoise	WhiteNoise
nTrials = 3
matrixSize = 16
SampleRate= 500
SimDuration= 1.0

CenterX = 4
CenterY = 4

SpatialFrequency = [0.25, 0.5, 0.75]
TemporalFrequency = [10, 20, 17.5] 
WaveDirection = 1  #lower or higher than 0 determines in or outward motion for targetWave and ClockWise or Counter ClockWise for RotatingWave 
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
    WaveDuration = WaveDuration, 
    CenterX = CenterX,
    CenterY = CenterY
    )

noise = SimulationFuns.simulate_signal("SpatialPinkNoise", nTrials, matrixSize, SampleRate, SimDuration)

SNR = 0.6
SimOutput = SimulationFuns.SNRMix(wave, noise, SNR)
SimOutput._simInfo.append({"SNRMix" : [{"NoiseType": "SpatialPinkNoise"}, {"SNR": SNR}, {"matrixSize": matrixSize}, {"sampleRate": SampleRate}]})
SimOutput.save_to_file("./ExampleData/WaveData_SIM_targetwaves_lowSNR")
filename ="./ExampleData/WaveData_SIM_targetwaves_lowSNR"
SimOutput.save_to_file(filename)
# save(SimOutput.get_data("SimulatedData"), filename + "_data")
# save(SimOutput.get_SimInfo(),filename + "_simInfo")

data = SimOutput.get_data("SimulatedData")
data = np.reshape(data ,(nTrials,matrixSize,matrixSize,int(np.floor(SampleRate*SimDuration))))
trialNr =2
fullstatus = data[trialNr] # Take single trial
print(SimOutput)
plt.imshow(fullstatus[:, :, 0], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.imshow(fullstatus[:, :, 200], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.plot(fullstatus[8,8,:])
plt.show()

#%% Stationary Pulse High SNR
Type =  "StationaryPulse" # PlaneWave	 StationaryPulse   TargetWave	RotatingWave	LocalOscillation	SpatialPinkNoise	WhiteNoise
nTrials = 3
matrixSize = 16
SampleRate= 500
SimDuration= 1.0

CenterX = 4
CenterY = 4
Sigma = 6

SpatialFrequency = [0.25, 0.5, 0.75]
TemporalFrequency = [10, 20, 17.5] 
WaveDirection = 1  #lower or higher than 0 determines in or outward motion for targetWave and ClockWise or Counter ClockWise for RotatingWave 
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
    WaveDirection = WaveDirection,
    TemporalFrequency = TemporalFrequency,
    CenterX = CenterX,
    CenterY = CenterY,
    Sigma=Sigma
    )

noise = SimulationFuns.simulate_signal("SpatialPinkNoise", nTrials, matrixSize, SampleRate, SimDuration)

SNR = 2.0
SimOutput = SimulationFuns.SNRMix(wave, noise, SNR)
SimOutput._simInfo.append({"SNRMix" : [{"NoiseType": "SpatialPinkNoise"}, {"SNR": SNR}, {"matrixSize": matrixSize}, {"sampleRate": SampleRate}]})
SimOutput.save_to_file("./ExampleData/WaveData_SIM_stationaryPulse_highSNR")
filename ="./ExampleData/WaveData_SIM_stationaryPulse_highSNR"
SimOutput.save_to_file(filename)
# save(SimOutput.get_data("SimulatedData"), filename + "_data")
# save(SimOutput.get_SimInfo(),filename + "_simInfo")

data = SimOutput.get_data("SimulatedData")
data = np.reshape(data ,(nTrials,matrixSize,matrixSize,int(np.floor(SampleRate*SimDuration))))
trialNr =2
fullstatus = data[trialNr] # Take single trial
print(SimOutput)
plt.imshow(fullstatus[:, :, 0], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.imshow(fullstatus[:, :, 200], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.plot(fullstatus[8,8,:])
plt.show()

#%% Stationary Pulse Low SNR
Type =  "StationaryPulse" # PlaneWave	 StationaryPulse   TargetWave	RotatingWave	LocalOscillation	SpatialPinkNoise	WhiteNoise
nTrials = 3
matrixSize = 16
SampleRate= 500
SimDuration= 1.0

CenterX = 4
CenterY = 4
Sigma = 5

SpatialFrequency = [0.25, 0.5, 0.75]
TemporalFrequency = [10, 20, 17.5] 
WaveDirection = 1  #lower or higher than 0 determines in or outward motion for targetWave and ClockWise or Counter ClockWise for RotatingWave 
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
    WaveDirection = WaveDirection,
    TemporalFrequency = TemporalFrequency,
    CenterX = CenterX,
    CenterY = CenterY,
    Sigma=Sigma
    )

noise = SimulationFuns.simulate_signal("SpatialPinkNoise", nTrials, matrixSize, SampleRate, SimDuration)

SNR = .6
SimOutput = SimulationFuns.SNRMix(wave, noise, SNR)
SimOutput._simInfo.append({"SNRMix" : [{"NoiseType": "SpatialPinkNoise"}, {"SNR": SNR}, {"matrixSize": matrixSize}, {"sampleRate": SampleRate}]})
SimOutput.save_to_file("./ExampleData/WaveData_SIM_stationaryPulse_lowSNR")
filename ="./ExampleData/WaveData_SIM_stationaryPulse_lowSNR"
SimOutput.save_to_file(filename)
# save(SimOutput.get_data("SimulatedData"), filename + "_data")
# save(SimOutput.get_SimInfo(),filename + "_simInfo")

data = SimOutput.get_data("SimulatedData")
data = np.reshape(data ,(nTrials,matrixSize,matrixSize,int(np.floor(SampleRate*SimDuration))))
trialNr =2
fullstatus = data[trialNr] # Take single trial
print(SimOutput)
plt.imshow(fullstatus[:, :, 0], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.imshow(fullstatus[:, :, 200], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.plot(fullstatus[8,8,:])
plt.show()

#%% Local Oscillators with a frequency gradient
Type =  "FrequencyGradient" # PlaneWave	 StationaryPulse   TargetWave	RotatingWave	LocalOscillation	SpatialPinkNoise	WhiteNoise
nTrials = 3
matrixSize = 20
SampleRate= 500
SimDuration= 3.0

MaxTemporalFrequency = 10
MinTemporalFrequency = 7.5

WaveDirection = [90,0,-30]  #lower or higher than 0 determines in or outward motion for targetWave and ClockWise or Counter ClockWise for RotatingWave 
#initialize data
onsetInMs = 300
#WaveOnset =  [(onsetInMs / (1000/SampleRate) ) / (1 / tf  * SampleRate)for tf in TemporalFrequency]
#WaveDuration = 10
wave = SimulationFuns.simulate_signal(
    Type, 
    nTrials, 
    matrixSize, 
    SampleRate, 
    SimDuration, 
    #SimOptions from here on
    MaxTemporalFrequency = MaxTemporalFrequency,
    MinTemporalFrequency = MinTemporalFrequency,
    WaveDirection = WaveDirection,
    )

noise = SimulationFuns.simulate_signal("SpatialPinkNoise", nTrials, matrixSize, SampleRate, SimDuration)

SNR = 2.0
SimOutput = SimulationFuns.SNRMix(wave, noise, SNR)
SimOutput._simInfo.append({"SNRMix" : [{"NoiseType": "SpatialPinkNoise"}, {"SNR": SNR}, {"matrixSize": matrixSize}, {"sampleRate": SampleRate}]})
filename ="./ExampleData/WaveData_SIM_freqgrad_onset300_highSNR"
SimOutput.save_to_file(filename)
# save(SimOutput.get_data("SimulatedData"), filename + "_data")
# save(SimOutput.get_SimInfo(),filename + "_simInfo")

data = SimOutput.get_data("SimulatedData")
data = np.reshape(data ,(nTrials,matrixSize,matrixSize,int(np.floor(SampleRate*SimDuration))))
trialNr =2
fullstatus = data[trialNr] # Take single trial
print(SimOutput)
plt.imshow(fullstatus[:, :, 0], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.imshow(fullstatus[:, :, 200], origin='lower',vmin=-1, vmax=1)
plt.show()
plt.plot(fullstatus[8,8,:])
plt.show()

hf.squareSpatialPositions(wave)
ani  = Plotting.animate_grid_data(wave, "SimulatedData", dataInd=2, probepositions=((0,0),(10,10),(15,15)))
ani.save("ExampleData/Output/TestWave.mp4")