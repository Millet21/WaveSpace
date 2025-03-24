#%%
from WaveSpace.Utils import ImportHelpers
from WaveSpace.SpatialArrangement import SensorLayout as sensors

import matplotlib.pyplot as plt
import pickle
#%%
#load data from file:
data= ImportHelpers.load_wavedata_object("ExampleData/WaveData_SIM_planewave_plus_localOscillators_highSNR")

#%%
#regular grid
#calculate sensor to sensor distance, where chanpos has x and y coordinates
pos = data.get_channel_positions() #get 2D coordinates of contacts
sensors.regularGrid(data, pos) #adds a distance matrix to the data object 

#plot
plt.imshow(data.get_distMat(), origin= 'lower')
plt.colorbar()
plt.title('Contact-to-Contact distance')
plt.xlabel('Contact')
plt.ylabel('Contact')


#%% Distance Matrix for regular grid of contacts
#project 3D coordinates to 2D space, preserving distanes between them as good as possible
sensors.distmat_to_2d_coordinates_MDS(data)
#plot 3D and 2D contact positions:
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data.get_channel_positions()[:,0], data.get_channel_positions()[:,1],data.get_channel_positions()[:,2])
plt.title('Contact position 3D ')
plt.figure()
plt.scatter(data.get_2d_coordinates()[:,0], data.get_2d_coordinates()[:,1])
plt.title('Contact position 2D embedding preserving inter-contact distances. Arbitrary units')

#%% cortical distance, warning: this is a slow operation

SurfaceFile = 'ExampleData/surfaceFileInflated_LH' #path to freesurfer generated cortical surface
with open(SurfaceFile, 'rb') as f:
    Surface = pickle.load(f)

sensors.distance_along_surface(data,Surface,tolerance=35)

plt.imshow(data.get_distMat(), origin= 'lower')
plt.colorbar()
plt.title('Contact-to-Contact distance along surface in mm')
plt.xlabel('SeedContact')
plt.ylabel('TargetContact')
