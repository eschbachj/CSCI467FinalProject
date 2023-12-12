import os
import cv2
import numpy as np
from skimage import io, exposure, color, data, restoration, measure
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, threshold_local
from skimage.segmentation import clear_border, mark_boundaries
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import jaccard_score

def cell_segmentation(image_path, correct_mask_path):
    # Load the DAPI staining image
    image = io.imread(image_path)
    
    # Convert to grayscale
    gray_image = rgb2gray(image)
    
    # Apply histogram equalization to enhance contrast
    enhanced_image = exposure.equalize_hist(gray_image)
    
    # Apply contrast stretching
    p2, p98 = np.percentile(gray_image, (2, 98))
    stretched_image = exposure.rescale_intensity(gray_image, in_range=(p2, p98))

    # Apply Otsu's thresholding to create a binary image
    threshold_value = threshold_otsu(stretched_image)
    binary_image = enhanced_image > threshold_value
    
    #Apply Gaussian thresholding to create a binary image
    #binary_image = threshold_local(stretched_image, block_size=31, method='gaussian', offset=0.01)
    
    # Remove small noise using morphological operations
    binary_image = closing(binary_image, square(3))
    
    # Label and measure segmented regions
    labeled_cells, num_cells = label(binary_image, background=0, return_num=True)
    cell_properties = regionprops(labeled_cells)
    
    # Filter out small objects (e.g., noise)
    min_cell_area = 100  # Adjust this threshold as needed
    segmented_cells = [cell for cell in cell_properties if cell.area >= min_cell_area]
    
    # Create a mask with the segmented cells
    cell_mask = np.zeros_like(binary_image)
    for cell in segmented_cells:
        for coordinates in cell.coords:
            cell_mask[coordinates[0], coordinates[1]] = 1
    
    inverted_cell_mask = 1-cell_mask

    # Ensure that the inverted cell mask is of sufficient size
    min_inverted_cell_area = 300  # Adjust this threshold as needed
    labeled_inverted_cells, num_inverted_cells = label(inverted_cell_mask, background=0, return_num=True)
    inverted_cell_properties = regionprops(labeled_inverted_cells)
    large_inverted_cells = [cell for cell in inverted_cell_properties if cell.area >= min_inverted_cell_area]

    # Create a mask with the segmented cells
    inv_cell_mask = np.zeros_like(binary_image)
    for cell in large_inverted_cells:
        for coordinates in cell.coords:
            inv_cell_mask[coordinates[0], coordinates[1]] = 1

    # Load the correct cell mask
    correct_mask = io.imread(correct_mask_path)
    
    return inv_cell_mask, correct_mask

# Process a whole test set
test_set_dir = './segmentation_dataset/final_datasets/test/images'
correct_masks_dir = './segmentation_dataset/test/masks'
output_dir = './segmentation_dataset/baseline_results_tmp'
subplots_output_dir = './segmentation_dataset/baseline_results/subplots_tmp'
os.makedirs(subplots_output_dir, exist_ok=True)
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

intersection_over_union_scores = []

subplot_fig = plt.figure(figsize=(15,5))
# Iterate through all image files in the test set directory
for filename in tqdm(os.listdir(test_set_dir)):
    if filename.endswith('.png'):
        image_path = os.path.join(test_set_dir, filename)
        correct_mask_filename = filename.replace('.png', '.png')
        correct_mask_path = os.path.join(correct_masks_dir, correct_mask_filename)
        
        # Extract the base filename (without extension)
        base_filename = os.path.splitext(filename)[0]
        
        # Perform cell segmentation and load correct mask
        cell_mask, correct_mask = cell_segmentation(image_path, correct_mask_path)
        
        # Compute Intersection over Union (IoU) score
        intersection = np.logical_and(cell_mask, correct_mask)
        union = np.logical_or(cell_mask, correct_mask)
        iou = np.sum(intersection) / np.sum(union)
        if iou>=0.3:
            print(base_filename)
        intersection_over_union_scores.append(iou)
        
        # Save the segmented image with cell boundaries
        original_image = io.imread(image_path)
        cell_mask = cell_mask.astype(np.uint8)
        segmented_image = mark_boundaries(original_image, cell_mask, color=(1, 0, 0))
        segmented_image = img_as_ubyte(segmented_image)  # Convert to uint8
        output_image_path = os.path.join(output_dir, filename)
        io.imsave(output_image_path, segmented_image)
        
        # Set the flag to False to indicate that it's no longer the first iteration
        first_iteration = False

        # Create a subplot image with original image, correct mask, and segmented cell mask
        plt.subplot(1, 3, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title("Original Image")

        plt.subplot(1, 3, 2)
        plt.imshow(correct_mask, cmap='gray')
        plt.title("Correct Mask")

        plt.subplot(1, 3, 3)
        plt.imshow(original_image, cmap='gray')
        plt.imshow(cell_mask, alpha=0.5, cmap='viridis')
        plt.title("Segmented Cells\nIoU: {:.4f}".format(iou))  # Display IoU score
        
        # Save the subplot image with the same name as the image file
        subplot_image_path = os.path.join(subplots_output_dir, f'{base_filename}_subplot.png')
        plt.savefig(subplot_image_path)
        plt.clf()  # Clear the figure after saving

# Close the figure at the end of the loop
plt.close(subplot_fig)
            
# Calculate and print the average IoU score
average_iou = np.mean(intersection_over_union_scores)
print(f"Average Intersection over Union (IoU) Score: {average_iou:.4f}")

print("Segmentation of the test set is complete.")