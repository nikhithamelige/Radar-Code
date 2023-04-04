import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
moving = 1
no_moving = 2

np.set_printoptions(threshold=np.inf)

frange_doppler_features = np.load("data/range_doppler_data.npz", allow_pickle=True)

x_data, y_data = frange_doppler_features['out_x'], frange_doppler_features['out_y']
# Config parameters for test
configParameters = {'numDopplerBins': 16, 'numRangeBins': 128, 'rangeResolutionMeters': 0.04360212053571429,
                    'rangeIdxToMeters': 0.04360212053571429, 'dopplerResolutionMps': 0.12518841691334906,
                    'maxRange': 10.045928571428572, 'maxVelocity': 2.003014670613585}

# Generate the range and doppler arrays for the plot
rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]
dopplerArray = np.multiply(np.arange(-configParameters["numDopplerBins"] / 2, configParameters["numDopplerBins"] / 2),
                           configParameters["dopplerResolutionMps"])

fig = plt.figure()

test = moving

for count, frame in enumerate(x_data[np.where(y_data == test)]):
    plt.clf()
    if test-1:
        plt.title(f"Frame {count} for no moving target/empty area")
    else:
        plt.title(f"Frame {count} for moving target")
    cs = plt.contourf(rangeArray, dopplerArray, frame)
    fig.colorbar(cs, shrink=0.9)
    fig.canvas.draw()
    plt.pause(0.1)

