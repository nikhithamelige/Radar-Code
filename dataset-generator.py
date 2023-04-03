import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isdir, join

dataset_path = 'data'

# Config parameters for test
configParameters = {'numDopplerBins': 16, 'numRangeBins': 128, 'rangeResolutionMeters': 0.04360212053571429,
                    'rangeIdxToMeters': 0.04360212053571429, 'dopplerResolutionMps': 0.12518841691334906,
                    'maxRange': 10.045928571428572, 'maxVelocity': 2.003014670613585}

all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
# all_targets.remove('.ipynb_checkpoints')

print(all_targets)

filenames = []
y = []

for index, target in enumerate(all_targets):
    filenames.append(listdir(join(dataset_path, target)))
    y.append(np.ones(len(filenames[index])) * index)


def calc_range_doppler(data_frame, packet_id, config):
    payload = data_frame[packet_id].to_numpy()
    # Convert levels to dBm
    payload = 20 * np.log10(payload)
    # Clac. range Doppler array
    rangeDoppler = np.reshape(payload, (config["numDopplerBins"], config["numRangeBins"]), 'F')
    rangeDoppler = np.append(rangeDoppler[int(len(rangeDoppler) / 2):], rangeDoppler[:int(len(rangeDoppler) / 2)],
                             axis=0)
    # calculate mag
    # row_sums = rangeDoppler.sum(axis=1)
    # gen normalised matrix
    # rangeDoppler = rangeDoppler / row_sums[:, np.newaxis]

    return rangeDoppler


out_x_rcs = []
out_y_rcs = []

for folder in range(len(all_targets)):
    all_files = join(dataset_path, all_targets[folder])
    for i in range(len(listdir(all_files))):
        full_path = join(all_files, listdir(all_files)[i])

        print(full_path, folder)

        df_data = pd.read_csv(full_path)

        for col in df_data.columns:
            data = calc_range_doppler(df_data, col, configParameters)
            out_x_rcs.append(data)
            out_y_rcs.append(folder + 1)

data_range_x = np.array(out_x_rcs)
data_range_y = np.array(out_y_rcs)

np.savez('data/range_doppler_data.npz', out_x=data_range_x, out_y=data_range_y)
