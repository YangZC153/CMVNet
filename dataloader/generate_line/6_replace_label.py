import SimpleITK as sitk
import numpy as np
import os

def check_and_relabel_nifti(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for file_name in os.listdir(input_directory):
        if file_name.endswith('_line.nii.gz'):
            input_file_path = os.path.join(input_directory, file_name)
            output_file_path = os.path.join(output_directory, file_name)

            image = sitk.ReadImage(input_file_path)
            img_array = sitk.GetArrayFromImage(image)

            unique_labels = np.unique(img_array)
            if not np.all(np.isin(unique_labels, [0, 1, 2])):
                raise ValueError(f"Unexpected label found in {file_name}. Expected labels 0, 1, 2.")

            img_array[img_array == 2] = 1

            relabeled_image = sitk.GetImageFromArray(img_array)
            relabeled_image.CopyInformation(image)

            sitk.WriteImage(relabeled_image, output_file_path)
            print(f"Processed file saved: {output_file_path}")

input_directory = "./data_4"
output_directory = "./data_5"

check_and_relabel_nifti(input_directory, output_directory)