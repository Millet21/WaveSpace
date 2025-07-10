# plots all conditions in SimulatedData as animations

import sys
import os

# Add the project root directory to the Python path when working with source code, 
# not necessary when package is installed
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path )
print(path)

from WaveSpace.PlottingHelpers import Plotting
from WaveSpace.Utils import ImportHelpers


dataPath  = os.path.join(path, "Examples/ExampleData/Output") 
waveData = ImportHelpers.load_wavedata_object(dataPath + "/SimulatedData")

#print(waveData.get_trialInfo())

trial_list = list(waveData.get_trialInfo())

for ind in range(0, len(trial_list), 2):
    print(f'animate {trial_list[ind]}')
    ani = Plotting.animate_grid_data(waveData, DataBucketName="SimulatedData", dataInd=ind, probepositions=[(0,15), (5,15), (10,15), (15,15), (19,15), (19,15)])
    plot_file = os.path.join(path, f"Examples/ExampleData/Output/{trial_list[ind]}_Animation.mp4")
    ani.save(plot_file)

