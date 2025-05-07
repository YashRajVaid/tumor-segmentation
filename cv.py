"""import cv2
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
def preprocess(image):
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)
    image_norm = cv2.normalize(image_blur, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return image_norm
def is_homogeneous(region, threshold_std=0.03):
    return np.std(region) < threshold_std
def split(image, x, y, w, h, threshold_std):
    regions = []
    region = image[y:y+h, x:x+w]
    if w <= 8 or h <= 8 or is_homogeneous(region, threshold_std):
        regions.append((x, y, w, h))
    else:
        half_w = w // 2
        half_h = h // 2
        regions += split(image, x, y, half_w, half_h, threshold_std)
        regions += split(image, x + half_w, y, w - half_w, half_h, threshold_std)
        regions += split(image, x, y + half_h, half_w, h - half_h, threshold_std)
        regions += split(image, x + half_w, y + half_h, w - half_w, h - half_h, threshold_std)

    return regions
def merge(image, regions, threshold_mean=0.1):
    mask = np.zeros(image.shape, np.uint8)
    for i, (x, y, w, h) in enumerate(regions):
        region = image[y:y+h, x:x+w]
        mean_val = np.mean(region)
        if mean_val > threshold_mean:
            mask[y:y+h, x:x+w] = 255
    return mask
def region_splitting_merging(image, threshold_std=0.05, threshold_mean=0.1):
    preprocessed = preprocess(image)
    height, width = preprocessed.shape
    regions = split(preprocessed, 0, 0, width, height, threshold_std)
    mask = merge(preprocessed, regions, threshold_mean)
    mask = morphology.opening(mask, morphology.disk(3))
    mask = morphology.closing(mask, morphology.disk(5))

    return mask

"""
# region_split_merge.py
import cv2
import numpy as np


def is_homogeneous(region, threshold):
    return np.std(region) < threshold


def split_region(image, x, y, w, h, threshold, min_size, regions):
    region = image[y:y+h, x:x+w]
    if w <= min_size or h <= min_size or is_homogeneous(region, threshold):
        regions.append((x, y, w, h, np.mean(region)))
        return

    half_w, half_h = w // 2, h // 2
    split_region(image, x, y, half_w, half_h, threshold, min_size, regions)
    split_region(image, x + half_w, y, w - half_w, half_h, threshold, min_size, regions)
    split_region(image, x, y + half_h, half_w, h - half_h, threshold, min_size, regions)
    split_region(image, x + half_w, y + half_h, w - half_w, h - half_h, threshold, min_size, regions)


def build_label_map(image_shape, regions):
    label_map = np.zeros(image_shape[:2], dtype=np.int32)
    labels = {}
    for idx, (x, y, w, h, val) in enumerate(regions, 1):
        label_map[y:y+h, x:x+w] = idx
        labels[idx] = val
    return label_map, labels


def get_neighbors(label_map, label):
    coords = np.argwhere(label_map == label)
    neighbors = set()
    for y, x in coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < label_map.shape[0] and 0 <= nx < label_map.shape[1]:
                    n_label = label_map[ny, nx]
                    if n_label != label and n_label != 0:
                        neighbors.add(n_label)
    return neighbors


def merge_regions(label_map, labels, threshold):
    merged = True
    while merged:
        merged = False
        new_labels = labels.copy()
        for label in list(labels.keys()):
            neighbors = get_neighbors(label_map, label)
            for n_label in neighbors:
                if abs(labels[label] - labels[n_label]) < threshold:
                    label_map[label_map == n_label] = label
                    merged = True
                    new_val = (labels[label] + labels[n_label]) / 2
                    new_labels[label] = new_val
                    del new_labels[n_label]
        labels = new_labels.copy()
    return label_map, labels


def create_segmented_image(label_map, labels):
    segmented = np.zeros(label_map.shape, dtype=np.uint8)
    for label, val in labels.items():
        segmented[label_map == label] = int(val)
    return segmented


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred


def segment_image(image, threshold=10, min_size=16, merge_threshold=15):
    preprocessed = preprocess_image(image)
    regions = []
    split_region(preprocessed, 0, 0, preprocessed.shape[1], preprocessed.shape[0], threshold, min_size, regions)
    label_map, labels = build_label_map(preprocessed.shape, regions)
    label_map, labels = merge_regions(label_map, labels, merge_threshold)
    segmented = create_segmented_image(label_map, labels)
    return segmented


"""
import numpy as np
import cv2

# Function to check if the region is homogeneous based on a threshold
def is_homogeneous(region, threshold):
    mean_intensity = np.mean(region)
    std_dev = np.std(region)
    return std_dev < threshold

# Region Splitting and Merging function
def region_split_merge(image, threshold, min_size, threshold_mean, threshold_std):

    # Helper function to split the region recursively
    def split_region(image, x, y, width, height):
        if width <= min_size or height <= min_size or is_homogeneous(image[y:y+height, x:x+width], threshold):
            return [(x, y, width, height)]  # Return the current region as a list

        # Recursively split the region into 4 quadrants
        mid_x = x + width // 2
        mid_y = y + height // 2

        regions = []
        regions += split_region(image, x, y, mid_x - x, mid_y - y)
        regions += split_region(image, mid_x, y, x + width - mid_x, mid_y - y)
        regions += split_region(image, x, mid_y, mid_x - x, y + height - mid_y)
        regions += split_region(image, mid_x, mid_y, x + width - mid_x, y + height - mid_y)

        return regions

    # Helper function to check if two regions should be merged based on intensity and texture
    def should_merge(region1, region2):
        x1, y1, w1, h1 = region1
        x2, y2, w2, h2 = region2

        # Calculate mean and std deviation for both regions
        mean1, std1 = np.mean(image[y1:y1+h1, x1:x1+w1]), np.std(image[y1:y1+h1, x1:x1+w1])
        mean2, std2 = np.mean(image[y2:y2+h2, x2:x2+w2]), np.std(image[y2:y2+h2, x2:x2+w2])

        # Check if both the mean and standard deviation differences are below thresholds
        mean_diff = abs(mean1 - mean2)
        std_diff = abs(std1 - std2)

        return mean_diff < threshold_mean and std_diff < threshold_std

    # Split the entire image into regions
    regions = split_region(image, 0, 0, image.shape[1], image.shape[0])

    # Merge adjacent regions if they meet the merging criteria
    merged_regions = []
    for i, region1 in enumerate(regions):
        merged = False
        for j, region2 in enumerate(regions):
            if i != j and should_merge(region1, region2):
                # Merge region1 and region2 by combining them into one larger region
                x1, y1, w1, h1 = region1
                x2, y2, w2, h2 = region2

                # Combine the bounding box of the two regions
                x = min(x1, x2)
                y = min(y1, y2)
                w = max(x1 + w1, x2 + w2) - x
                h = max(y1 + h1, y2 + h2) - y

                merged_regions.append((x, y, w, h))
                merged = True
                break

        if not merged:
            merged_regions.append(region1)

    # Create an empty label image for segmentation
    segmented_image = np.zeros(image.shape, dtype=np.uint8)

    # Label the regions in the segmented image
    label = 1
    for region in merged_regions:
        x, y, w, h = region
        segmented_image[y:y+h, x:x+w] = label
        label += 1

    return segmented_image

"""