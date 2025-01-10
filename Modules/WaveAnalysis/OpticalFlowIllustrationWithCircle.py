import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from Modules.Utils import WaveData as wd, HelperFuns as hf
from Modules.Utils import ImportHelpers
from Modules.WaveAnalysis import OpticalFlow

# Parameters
frame_size = (640, 480)
circle_radius = 40
num_frames = 50
frames = []

# Initialize the first frame with a circle at the center
center_pos = (frame_size[0] // 2, frame_size[1] // 2)
circle_pos = center_pos

# Function to generate frames
def generate_frames():
    global circle_pos
    # Move to the upper left
    for _ in range(num_frames // 4):
        frame = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255  # White background
        circle_pos = (circle_pos[0] - 10, circle_pos[1] - 15)
        cv2.circle(frame, circle_pos, circle_radius, (0, 0, 255), -1)  # Red circle
        frames.append(frame)

    # Move to the lower right
    for _ in range(num_frames // 4):
        frame = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255  # White background
        circle_pos = (circle_pos[0] + 15, circle_pos[1] + 15)
        cv2.circle(frame, circle_pos, circle_radius, (0, 0, 255), -1)  # Red circle
        frames.append(frame)

    # Move in a large circle around the center, clockwise
    for angle in np.linspace(0, 2 * np.pi, num_frames // 2):
        frame = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255  # White background
        circle_pos = (int(center_pos[0] + 100 * np.cos(angle)),
                      int(center_pos[1] + 100 * np.sin(angle)))
        cv2.circle(frame, circle_pos, circle_radius, (0, 0, 255), -1)  # Red circle
        frames.append(frame)

    # Move in a large circle around the center, counterclockwise
    for angle in np.linspace(2 * np.pi, 0, num_frames // 2):
        frame = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255  # White background
        circle_pos = (int(center_pos[0] + 100 * np.cos(angle)),
                      int(center_pos[1] + 100 * np.sin(angle)))
        cv2.circle(frame, circle_pos, circle_radius, (0, 0, 255), -1)  # Red circle
        frames.append(frame)

# Generate the frames
generate_frames()

# Convert frames to grayscale and store in a NumPy array
bw_frames = np.zeros((frame_size[1], frame_size[0], len(frames)), dtype=np.uint8)
for i, frame in enumerate(frames):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bw_frames[:, :, i] = gray_frame
bw_frames = bw_frames[::5, ::5, :]
bw_frames = bw_frames + 0.001

waveData = ImportHelpers.load_wavedata_object('OpticalFlow_illustrationWithCircle')
data = np.expand_dims(bw_frames, axis=0)
waveData.DataBuckets['SimulatedData'].set_data(data, 'trl_posx_posy_time')

OpticalFlow.create_uv(waveData, 
        applyGaussianBlur=False, 
        dataBucketName= 'SimulatedData',
        type = "real", 
        Sigma=0, 
        alpha = 0.01, 
        nIter = 200, 
        is_phase = False)

plt.quiver(np.real(waveData.get_data('UV')[0,:,:,50]) , np.imag(waveData.get_data('UV')[0,:,:,50]))

uv_data = waveData.get_data('UV')[0,:,:,:]
#plot with flow vectors
def update(i):
    ax.clear()
    frame = bw_frames[:, :, i]
    ax.imshow(frame, cmap='gray', interpolation='antialiased')  # Apply anti-aliasing
    if i < uv_data.shape[2]:
        u = np.real(uv_data[:,:,i])
        v = np.imag(uv_data[:,:,i])
        magnitude = np.sqrt(u**2 + v**2)
        mask = magnitude > 1.5
        y, x = np.meshgrid(np.arange(u.shape[0]), np.arange(u.shape[1]), indexing='ij')
        ax.quiver(x[mask], y[mask], u[mask], v[mask], color='r', scale=30)  # Black flow vectors
    ax.axis('off')

fig, ax = plt.subplots(dpi=150)  # Increase DPI for higher resolution
fig.patch.set_facecolor('white')  # Set figure background to white
ani = plt.matplotlib.animation.FuncAnimation(fig, update, frames=len(bw_frames[0, 0, :]), repeat=False)
ani.save('moving_circle_with_flow.gif', writer=PillowWriter(fps=5))

#plot without flow vectors
def update(i):
    ax.clear()
    frame = bw_frames[:, :, i]
    ax.imshow(frame, cmap='gray', interpolation='antialiased')  # Apply anti-aliasing
    if i < uv_data.shape[2]:
        u = np.real(uv_data[:,:,i])
        v = np.imag(uv_data[:,:,i])
        magnitude = np.sqrt(u**2 + v**2)
        mask = magnitude > 50
        y, x = np.meshgrid(np.arange(u.shape[0]), np.arange(u.shape[1]), indexing='ij')
        ax.quiver(x[mask], y[mask], u[mask], v[mask], color='r', scale=50)  # Black flow vectors
    ax.axis('off')

fig, ax = plt.subplots(dpi=150)  # Increase DPI for higher resolution
fig.patch.set_facecolor('white')  # Set figure background to white
ani = plt.matplotlib.animation.FuncAnimation(fig, update, frames=len(bw_frames[0, 0, :]), repeat=False)
ani.save('moving_circle.gif', writer=PillowWriter(fps=5))

