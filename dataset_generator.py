import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class NoiseMeasurementDataset(Dataset):
    def __init__(self, root_dir, N=300):
        """
        Args:
            root_dir (string): Directory with all the experiments.
            N (int): The size of the NxN grid for the observer noise data.
        """
        self.root_dir = root_dir
        self.N = N
        self.data_files = self._gather_files()

    def _gather_files(self):
        """
        Gather all files in the directory structure.
        """
        data_files = []
        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.isdir(folder_path):
                # Extract Vcruise and RPM from folder name
                vcr = int(folder_name.split('_vcr_')[0])
                rpm = int(folder_name.split('_RPM')[0].split('_')[-1])
                
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".h5"):
                        # Extract height from file name
                        height = int(file_name.split('dB_')[1].split('m_height')[0])
                        # Full path to .h5 file
                        file_path = os.path.join(folder_path, file_name)
                        data_files.append((vcr, rpm, height, file_path))
        return data_files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        # Extract Vcruise, RPM, height, and file path from the gathered data
        vcruise, rpm, height, file_path = self.data_files[index]

        # Load noise data from the .h5 file
        with h5py.File(file_path, 'r') as f:
            # Assuming the noise data is saved as a 300x300x750 dataset within the file
            noise_data = np.array(f['dataset'])  # Replace 'dataset' with the actual dataset name in the file

        # Average over the timestamps to get a 300x300 grid for each case
        max_dBA = np.max(noise_data, axis=2)  # Max dBA across time, resulting in a 300x300 grid

        # Convert to PyTorch tensors
        input_data = torch.tensor([vcruise, rpm, height], dtype=torch.float32)
        output_data = torch.tensor(max_dBA, dtype=torch.float32)

        return input_data, output_data

# Instantiate the dataset and dataloader
root_dir = '/path/to/your/data/'  # Replace with your root directory
dataset = NoiseMeasurementDataset(root_dir=root_dir, N=300)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example to check if the dataloader works
for inputs, targets in dataloader:
    print("Inputs (Vcruise, RPM, height):", inputs)
    print("Targets (300x300 max dBA grid):", targets)
    break
