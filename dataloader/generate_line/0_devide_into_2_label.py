import SimpleITK as sitk
import numpy as np
import os


def get_unique_labels(label_image):
    label_stats = sitk.LabelStatisticsImageFilter()
    label_stats.Execute(label_image, label_image)

    unique_labels = label_stats.GetLabels()
    return unique_labels


def calculate_centroid(label_image, label):
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(label_image)
    centroid = statistics.GetCentroid(label)
    return np.array(centroid)


def binarize_image(image, threshold=0.5):
    image_array = sitk.GetArrayFromImage(image)
    min_pixel_value = np.min(image_array)
    max_pixel_value = np.max(image_array)

    if min_pixel_value == 0 and max_pixel_value == 1:
        return image

    binary_filter = sitk.BinaryThresholdImageFilter()
    lower_threshold = float(min_pixel_value + threshold * (max_pixel_value - min_pixel_value))
    upper_threshold = float(max_pixel_value)
    binary_filter.SetLowerThreshold(lower_threshold)
    binary_filter.SetUpperThreshold(upper_threshold)
    binary_filter.SetInsideValue(1)
    binary_filter.SetOutsideValue(0)
    binary_image = binary_filter.Execute(image)

    binary_image_array = sitk.GetArrayFromImage(binary_image)
    unique_values = np.unique(binary_image_array)

    return binary_image


def apply_morphological_opening(binary_image, radius):
    opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
    opening_filter.SetKernelType(sitk.sitkBall)
    opening_filter.SetKernelRadius(radius)
    opened_image = opening_filter.Execute(binary_image)
    return opened_image


def remove_small_objects(label_image, size_threshold):
    cc_filter = sitk.ConnectedComponentImageFilter()
    labeled_image = cc_filter.Execute(label_image)
    
    relabel_filter = sitk.RelabelComponentImageFilter()
    relabel_filter.SetMinimumObjectSize(size_threshold)
    relabeled_image = relabel_filter.Execute(labeled_image)
    
    return relabeled_image


def merge_closest_components(label_image):
    binary_image = binarize_image(label_image)

    cleaned_image = apply_morphological_opening(binary_image, radius=2)
    cleaned_image = remove_small_objects(cleaned_image, size_threshold=1380)

    cc_filter = sitk.ConnectedComponentImageFilter()
    label_image = cc_filter.Execute(cleaned_image > 0)
    n_components = cc_filter.GetObjectCount()

    centroids = [calculate_centroid(label_image, i+1) for i in range(n_components)]

    while n_components > 2:
        min_distance = float('inf')
        pair_to_merge = (0, 0)

        for i in range(n_components):
            for j in range(i + 1, n_components):
                distance = np.linalg.norm(centroids[i] - centroids[j])
                if distance < min_distance:
                    min_distance = distance
                    pair_to_merge = (i, j)

        label_map = {k+1: k+1 for k in range(n_components)}
        merged_label = pair_to_merge[0]+1
        label_to_remove = pair_to_merge[1]+1
        label_map[label_to_remove] = merged_label

        change_filter = sitk.ChangeLabelImageFilter()
        change_filter.SetChangeMap(label_map)
        label_image = change_filter.Execute(label_image)

        relabel_filter = sitk.RelabelComponentImageFilter()
        label_image = relabel_filter.Execute(label_image)

        n_components = relabel_filter.GetNumberOfObjects()
        centroids = [calculate_centroid(label_image, i+1) for i in range(n_components)]

    return label_image


def relabel_components(label_image):
    print("label_image", label_image.GetSize())
    unique_labels = get_unique_labels(label_image)
    unique_labels = [label for label in unique_labels if label != 0]

    if len(unique_labels) != 2:
        raise ValueError("Image does not have exactly two components. Found: {}".format(unique_labels))

    centroids = [calculate_centroid(label_image, label) for label in unique_labels]

    x_diff = np.abs(centroids[0][0] - centroids[1][0])
    y_diff = np.abs(centroids[0][1] - centroids[1][1])
    z_diff = np.abs(centroids[0][2] - centroids[1][2])
    print(f"X diff: {x_diff}, Y diff: {y_diff}, Z diff: {z_diff}")

    x_positions = [centroid[0] for centroid in centroids]

    left_label = unique_labels[0] if x_positions[0] < x_positions[1] else unique_labels[1]
    right_label = unique_labels[1] if left_label == unique_labels[0] else unique_labels[0]

    change_filter = sitk.ChangeLabelImageFilter()
    label_map = {left_label: 1, right_label: 2}
    change_filter.SetChangeMap(label_map)
    return change_filter.Execute(label_image)


def process_image(file_path):
    image = sitk.ReadImage(file_path)
    image = ensure_uint8(image)
    merged_image = merge_closest_components(image)
    print(file_path)
    relabeled_image = relabel_components(merged_image)
    return relabeled_image


def process_directory(input_directory, output_directory):
    for file_name in os.listdir(input_directory):
        if file_name.endswith('.nii.gz'):
            file_path = os.path.join(input_directory, file_name)
            processed_image = process_image(file_path)

            output_path = os.path.join(output_directory, file_name)
            sitk.WriteImage(processed_image, output_path)


def ensure_uint8(binary_image):
    if binary_image.GetPixelID() != sitk.sitkUInt8:
        binary_image = sitk.Cast(binary_image, sitk.sitkUInt8)
    return binary_image


input_directory = "./data_0"
output_directory = "./data_1"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

process_directory(input_directory, output_directory)

print("Result:" + output_directory)