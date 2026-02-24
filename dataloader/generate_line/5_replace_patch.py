import SimpleITK as sitk
import numpy as np
import os
import json

def read_bboxes(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

def numpy_paste(original_array, segmented_array, start_coords, original_size):
    end_coords = [min(start_coords[i] + segmented_array.shape[i], original_size[i]) for i in range(3)]

    original_slice = tuple(slice(max(start_coords[i], 0), end_coords[i]) for i in range(3))
    segmented_slice = tuple(slice(0, end_coords[i] - start_coords[i]) for i in range(3))

    original_array[original_slice] = segmented_array[segmented_slice]

    return original_array

def process_files(input_directory, left_directory, right_directory, bboxes_data, output_directory):
    for file_name in os.listdir(left_directory):
        if file_name.endswith('_left_line.nii.gz'):
            base_name = file_name.replace('_left_line.nii.gz', '.nii.gz')

            if base_name not in bboxes_data:
                print(f"File not found in bounding box data: {base_name}")
                continue

            original_image_path = os.path.join(input_directory, base_name)
            original_image = sitk.ReadImage(original_image_path)
            original_array = sitk.GetArrayFromImage(original_image)
            original_size = original_image.GetSize()

            left_image_path = os.path.join(left_directory, file_name)
            right_image_path = os.path.join(right_directory, base_name.replace('.nii.gz', '_right_line.nii.gz'))

            left_image = sitk.ReadImage(left_image_path)
            right_image = sitk.ReadImage(right_image_path)

            left_array = sitk.GetArrayFromImage(left_image)
            right_array = sitk.GetArrayFromImage(right_image)

            left_start_np = (bboxes_data[base_name]['Left_IAN']['z_min'], bboxes_data[base_name]['Left_IAN']['y_min'], bboxes_data[base_name]['Left_IAN']['x_min'])
            right_start_np = (bboxes_data[base_name]['Right_IAN']['z_min'], bboxes_data[base_name]['Right_IAN']['y_min'], bboxes_data[base_name]['Right_IAN']['x_min'])

            original_array = numpy_paste(original_array, left_array, left_start_np, original_size[::-1])
            original_array = numpy_paste(original_array, right_array, right_start_np, original_size[::-1])

            output_image = sitk.GetImageFromArray(original_array)
            output_image.CopyInformation(original_image)

            output_path = os.path.join(output_directory, base_name.replace('.nii.gz', '_line.nii.gz'))
            sitk.WriteImage(output_image, output_path)

input_directory = "./data_1"
left_directory = "./data_3/data_left"
right_directory = "./data_3/data_right"
output_directory = "./data_4"

bboxes_data = read_bboxes("./IAN_bboxes.json")

os.makedirs(output_directory, exist_ok=True)

process_files(input_directory, left_directory, right_directory, bboxes_data, output_directory)