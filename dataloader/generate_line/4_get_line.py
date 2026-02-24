from scipy.interpolate import CubicSpline
import cv2
import numpy as np
import SimpleITK as sitk
import os

def save_interpolated_curve_as_nifti(img_array, output_path):
    interpolated_image = sitk.GetImageFromArray(img_array)
    sitk.WriteImage(interpolated_image, output_path)

def detect_label_and_spatial_info(image):
    img_array = sitk.GetArrayFromImage(image)
    max_label = np.max(img_array)
    labeled_voxels = np.argwhere(img_array == max_label)
    start_pos = labeled_voxels.min(axis=0)
    end_pos = labeled_voxels.max(axis=0)
    dimensions = end_pos - start_pos + 1
    return max_label, dimensions

def compute_center_of_mass(image_slice, label_value):
    labeled_voxels = np.argwhere(image_slice == label_value)
    if labeled_voxels.size == 0:
        return np.nan, np.nan
    center_of_mass = labeled_voxels.mean(axis=0)
    center_of_mass = np.floor(center_of_mass).astype(int)
    return tuple(center_of_mass)

def extract_slices_and_analyze(image, max_label, dimensions, num_slices):
    img_array = sitk.GetArrayFromImage(image)
    y_coords_initial = np.linspace(1, dimensions[1] - 2, num_slices)

    slice_coordinates = []

    first_y = int(y_coords_initial[0])
    while np.all(img_array[:, first_y, :] == 0) and first_y > 0:
        first_y -= 1

    last_y = int(y_coords_initial[-1])
    while np.all(img_array[:, last_y, :] == 0) and last_y < dimensions[1] - 1:
        last_y += 1

    y_coords = np.linspace(first_y, last_y, num_slices)

    for y in y_coords:
        image_slice = img_array[:, int(y), :]
        if np.all(image_slice == 0):
            slice_coordinates.append((np.nan, y, np.nan))
        else:
            x_center, z_center = compute_center_of_mass(image_slice, max_label)
            slice_coordinates.append((x_center, y, z_center))

    return slice_coordinates

def interpolate_coordinates(coordinates, num_points=1000000):
    t = np.linspace(0, 1, len(coordinates))
    t_fine = np.linspace(0, 1, num_points)

    x, y, z = coordinates.T
    valid = ~np.isnan(x)

    cs_x = CubicSpline(t[valid], x[valid])
    cs_y = CubicSpline(t[valid], y[valid])
    cs_z = CubicSpline(t[valid], z[valid])

    x_interpolated = cs_x(t_fine)
    y_interpolated = cs_y(t_fine)
    z_interpolated = cs_z(t_fine)

    return np.vstack((x_interpolated, y_interpolated, z_interpolated)).T

def post_process_coordinates(coordinates):
    rounded_coordinates = np.round(coordinates).astype(int)
    unique_coordinates = np.unique(rounded_coordinates, axis=0)
    return unique_coordinates


def post_process_and_save_image(coordinates, img_size, max_label, output_path, dilation_radius=1):
    image_array = np.zeros(img_size, dtype=np.uint8)

    depth, height, width = img_size

    for coord in coordinates:
        z, y, x = coord

        if 0 <= z < depth and 0 <= y < height and 0 <= x < width:
            image_array[z, y, x] = max_label

    for y in range(image_array.shape[1]):
        slice_xz = image_array[:, y, :]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation_radius + 1, 1))

        dilated_slice_xz = cv2.dilate(slice_xz, kernel, iterations=1)

        image_array[:, y, :] = dilated_slice_xz

    dilated_image_sitk = sitk.GetImageFromArray(image_array)

    sitk.WriteImage(dilated_image_sitk, output_path)


def process_image(input_file_path, output_file_path, num_slices=10):
    reader = sitk.ImageFileReader()
    reader.SetFileName(input_file_path)
    image = reader.Execute()
    img_size = image.GetSize()

    max_label, dimensions = detect_label_and_spatial_info(image)
    coordinates = extract_slices_and_analyze(image, max_label, dimensions, num_slices)

    interpolated_coordinates = interpolate_coordinates(np.array(coordinates))
    processed_coordinates = post_process_coordinates(interpolated_coordinates)

    post_process_and_save_image(processed_coordinates, img_size, max_label, output_file_path)


def process_files_in_directory(input_dir, output_dir):
    for subdir, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.nii.gz'):
                input_file_path = os.path.join(subdir, file)
                relative_path = os.path.relpath(subdir, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                new_file_name = file.replace('.nii.gz', '_line.nii.gz')
                output_file_path = os.path.join(output_subdir, new_file_name)
                print(f'Processing {input_file_path} -> {output_file_path}')
                process_image(input_file_path, output_file_path)


def main():
    input_directory = './data_2'
    output_directory = './data_3'
    process_files_in_directory(input_directory, output_directory)


if __name__ == "__main__":
    main()