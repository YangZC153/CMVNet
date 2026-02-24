import json


def find_largest_dimensions(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    max_x_size = max_y_size = max_z_size = 0
    max_x_bbox = max_y_bbox = max_z_bbox = None
    max_x_file = max_y_file = max_z_file = None

    for file_name, bbox_data in data.items():
        for side, bbox in bbox_data.items():
            if side in ['Left_IAN', 'Right_IAN']:
                x_size, y_size, z_size = bbox['x_size'], bbox['y_size'], bbox['z_size']

                if x_size > max_x_size:
                    max_x_size = x_size
                    max_x_bbox = bbox
                    max_x_file = file_name + " (" + side + ")"

                if y_size > max_y_size:
                    max_y_size = y_size
                    max_y_bbox = bbox
                    max_y_file = file_name + " (" + side + ")"

                if z_size > max_z_size:
                    max_z_size = z_size
                    max_z_bbox = bbox
                    max_z_file = file_name + " (" + side + ")"

    return max_x_file, max_x_bbox, max_y_file, max_y_bbox, max_z_file, max_z_bbox


def create_combined_max_bbox(max_x_bbox, max_y_bbox, max_z_bbox):
    combined_bbox = {
        'x_size': max(max_x_bbox['x_size'], max_y_bbox['x_size'], max_z_bbox['x_size']),
        'y_size': max(max_x_bbox['y_size'], max_y_bbox['y_size'], max_z_bbox['y_size']),
        'z_size': max(max_x_bbox['z_size'], max_y_bbox['z_size'], max_z_bbox['z_size'])
    }
    return combined_bbox


json_file_path = "./IAN_bboxes.json"
max_x_file, max_x_bbox, max_y_file, max_y_bbox, max_z_file, max_z_bbox = find_largest_dimensions(json_file_path)
print(f"Largest X size is in file: {max_x_file}, BBOX: {max_x_bbox}")
print(f"Largest Y size is in file: {max_y_file}, BBOX: {max_y_bbox}")
print(f"Largest Z size is in file: {max_z_file}, BBOX: {max_z_bbox}")


combined_max_bbox = create_combined_max_bbox(max_x_bbox, max_y_bbox, max_z_bbox)
print(f"Combined Max BBOX: {combined_max_bbox}")

max_of_x_z = max(combined_max_bbox['x_size'], combined_max_bbox['z_size'])
propose_bbox = (max_of_x_z, combined_max_bbox['y_size'], max_of_x_z)
print(f"Proposed BBox size: {propose_bbox}")