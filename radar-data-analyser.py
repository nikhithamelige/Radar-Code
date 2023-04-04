import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

range_data_path, test = "data/occupied_room/occupied_room_t.csv", 1  # occupied
# range_data_path, test = "data/empty_room/empty_room_t.csv", 2  # empty

df = pd.read_csv(range_data_path, index_col=False)

# Config parameters for test
configParameters = {'numDopplerBins': 16, 'numRangeBins': 128, 'rangeResolutionMeters': 0.04360212053571429,
                    'rangeIdxToMeters': 0.04360212053571429, 'dopplerResolutionMps': 0.12518841691334906,
                    'maxRange': 10.045928571428572, 'maxVelocity': 2.003014670613585}

# Generate the range and doppler arrays for the plot
rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]
dopplerArray = np.multiply(np.arange(-configParameters["numDopplerBins"] / 2, configParameters["numDopplerBins"] / 2),
                           configParameters["dopplerResolutionMps"])


def calculate_range_doppler_heatmap(payload, config):
    # Convert levels to dBm
    payload = 20 * np.log10(payload)
    # Calc. range Doppler array
    rangeDoppler = np.reshape(payload, (config["numDopplerBins"], config["numRangeBins"]), 'F')
    rangeDoppler = np.append(rangeDoppler[int(len(rangeDoppler) / 2):], rangeDoppler[:int(len(rangeDoppler) / 2)],
                             axis=0)
    return rangeDoppler


def apply_2d_cfar(signal, guard_band_width, kernel_size, threshold_factor):
    num_rows, num_cols = signal.shape
    thresholded_signal = np.zeros((num_rows, num_cols))
    for i in range(guard_band_width, num_rows - guard_band_width):
        for j in range(guard_band_width, num_cols - guard_band_width):
            # Estimate the noise level
            noise_level = np.mean(np.concatenate((
                signal[i - guard_band_width:i + guard_band_width, j - guard_band_width:j + guard_band_width].ravel(),
                signal[i - kernel_size:i + kernel_size, j - kernel_size:j + kernel_size].ravel())))
            # Calculate the threshold for detection
            threshold = threshold_factor * noise_level
            # Check if the signal exceeds the threshold
            if signal[i, j] > threshold:
                thresholded_signal[i, j] = 1
    return thresholded_signal


guard_band_width = 3
kernel_size = 3
threshold_factor = 1

fig = plt.figure()

for count, frame in enumerate(df.columns):
    doppler_data = df[frame].to_numpy()
    range_doppler = calculate_range_doppler_heatmap(doppler_data, configParameters)
    # range_doppler = apply_2d_cfar(range_doppler, guard_band_width, kernel_size, threshold_factor)
    plt.clf()
    if test - 1:
        plt.title(f"Frame {count} for no moving target/empty area")
    else:
        plt.title(f"Frame {count} for moving target")
    cs = plt.contourf(rangeArray, dopplerArray, range_doppler)
    fig.colorbar(cs, shrink=0.9)
    fig.canvas.draw()
    plt.pause(0.1)


