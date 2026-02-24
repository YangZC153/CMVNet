import SimpleITK as sitk
import os
import json


def read_bboxes(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data


def crop_and_save_image(image, start_coords, bbox_size, output_path):
    image_size = image.GetSize()

    cropped_image = sitk.Image([bbox_size['x_size'], bbox_size['y_size'], bbox_size['z_size']], image.GetPixelID())

    cropped_image = cropped_image * 0

    crop_x_start = max(start_coords[0], 0)
    crop_y_start = max(start_coords[1], 0)
    crop_z_start = max(start_coords[2], 0)
    crop_x_end = min(start_coords[0] + bbox_size['x_size'], image_size[0])
    crop_y_end = min(start_coords[1] + bbox_size['y_size'], image_size[1])
    crop_z_end = min(start_coords[2] + bbox_size['z_size'], image_size[2])

    paste_x_start = max(0, -start_coords[0])
    paste_y_start = max(0, -start_coords[1])
    paste_z_start = max(0, -start_coords[2])

    extracted_region = sitk.RegionOfInterest(image, (crop_x_end - crop_x_start, crop_y_end - crop_y_start, crop_z_end - crop_z_start), (crop_x_start, crop_y_start, crop_z_start))

    cropped_image = sitk.Paste(cropped_image, extracted_region, extracted_region.GetSize(), (paste_x_start, paste_y_start, paste_z_start), (0, 0, 0))

    sitk.WriteImage(cropped_image, output_path)


def process_files(input_directory, bboxes_data, output_directory_left, output_directory_right, bbox_size):
    for file_name in os.listdir(input_directory):
        if file_name.endswith('.nii.gz'):
            file_path = os.path.join(input_directory, file_name)
            image = sitk.ReadImage(file_path)

            base_name, ext = os.path.splitext(file_name)
            if ext == ".gz":
                base_name, ext = os.path.splitext(base_name)
                ext += ".gz"

            left_output_name = f"{base_name}_left{ext}"
            right_output_name = f"{base_name}_right{ext}"

            left_ian_start = (bboxes_data[file_name]['Left_IAN']['x_min'], bboxes_data[file_name]['Left_IAN']['y_min'], bboxes_data[file_name]['Left_IAN']['z_min'])
            right_ian_start = (bboxes_data[file_name]['Right_IAN']['x_min'], bboxes_data[file_name]['Right_IAN']['y_min'], bboxes_data[file_name]['Right_IAN']['z_min'])

            crop_and_save_image(image, left_ian_start, bbox_size, os.path.join(output_directory_left, left_output_name))
            crop_and_save_image(image, right_ian_start, bbox_size, os.path.join(output_directory_right, right_output_name))
            print("Done!")


bbox_size = {'x_size': 194, 'y_size': 306, 'z_size': 194}

bboxes_data = read_bboxes("./IAN_bboxes.json")

input_directory = "./data_1"
output_directory_left = "./data_2/data_left"
output_directory_right = "./data_2/data_right"

os.makedirs(output_directory_left, exist_ok=True)
os.makedirs(output_directory_right, exist_ok=True)

process_files(input_directory, bboxes_data, output_directory_left, output_directory_right, bbox_size)