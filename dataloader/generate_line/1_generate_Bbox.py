import SimpleITK as sitk
import os
import json

def calculate_bounding_box(label_image, label):
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(label_image)
    bounding_box = label_shape_filter.GetBoundingBox(label)
    return bounding_box

def convert_bbox_to_dict(bbox):
    return {
        'x_min': bbox[0],
        'y_min': bbox[1],
        'z_min': bbox[2],
        'x_size': bbox[3],
        'y_size': bbox[4],
        'z_size': bbox[5]
    }

def process_files_and_generate_json(input_directory, output_json_path):
    data = {}

    for file_name in os.listdir(input_directory):
        if file_name.endswith('.nii.gz'):
            file_path = os.path.join(input_directory, file_name)
            label_image = sitk.ReadImage(file_path)

            image_size = label_image.GetSize()

            bbox1 = calculate_bounding_box(label_image, 1)
            bbox2 = calculate_bounding_box(label_image, 2)

            bbox1_dict = convert_bbox_to_dict(bbox1)
            bbox2_dict = convert_bbox_to_dict(bbox2)

            data[file_name] = {
                'Image_Size': image_size,
                'Left_IAN': bbox1_dict,
                'Right_IAN': bbox2_dict
            }

    with open(output_json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

input_directory = "./data_1"
output_json_path = "./IAN_bboxes.json"

process_files_and_generate_json(input_directory, output_json_path)