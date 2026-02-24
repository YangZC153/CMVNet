import SimpleITK as sitk
import numpy as np
import os

def convert_nii_to_npy(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for filename in os.listdir(input_directory):
        if filename.endswith('.nii.gz'):
            file_path = os.path.join(input_directory, filename)
            sitk_image = sitk.ReadImage(file_path)
            np_array = sitk.GetArrayFromImage(sitk_image)
            npy_filename = filename.replace('.nii.gz', '.npy')
            output_path = os.path.join(output_directory, npy_filename)
            np.save(output_path, np_array)
            print(f"Saved {output_path}")

input_directory = "./data_5"
output_directory = "./data_6"

convert_nii_to_npy(input_directory, output_directory)