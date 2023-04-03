import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

np.set_printoptions(threshold=np.inf)

frange_doppler_features = np.load("data/range_doppler_data.npz", allow_pickle=True)

x_data, y_data = frange_doppler_features['out_x'], frange_doppler_features['out_y']

fig = plt.figure()


for count, frame in enumerate(x_data[np.where(y_data == 1)]):
    plt.clf()
    plt.title(f"Frame {count} for no moving target/empty area")
    cs = plt.contourf(frame)
    fig.colorbar(cs, shrink=0.9)
    fig.canvas.draw()
    plt.pause(0.1)

