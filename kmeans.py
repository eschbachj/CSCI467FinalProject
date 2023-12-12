import os
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the number of clusters (k) for K-means
num_clusters = 2

# Specify the paths for your datasets, masks, and output directory
train_data_dir = './segmentation_dataset/final_datasets/train_set'
dev_data_dir = './segmentation_dataset/final_datasets/dev_set'
test_data_dir = './segmentation_dataset/final_datasets/test/images'

train_mask_dir = './segmentation_dataset/final_datasets/dev_train_masks'
dev_mask_dir = './segmentation_dataset/final_datasets/dev_train_masks'
test_mask_dir = './segmentation_dataset/final_datasets/test/masks'

output_dir = "./kmeans_output"

# Function to load and preprocess data
file_names = []
def load_and_preprocess_data(data_dir, mask_dir, flag):
    images = []
    masks = []

    for filename in tqdm(os.listdir(data_dir), desc="Loading and preprocessing"):
        if filename.endswith(".png"):  # Assuming your data is in PNG format
            # Load corresponding mask (ground truth) from the mask directory
            mask_filename = os.path.join(mask_dir, filename)

            if os.path.exists(mask_filename):
                image = cv2.imread(os.path.join(data_dir, filename))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                images.append(gray)

                mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
                masks.append(mask)
                
                if flag==1:
                    file_names.append(filename)
    return images, masks

# Function to calculate Intersection over Union (IoU)
def calculate_iou(predicted_mask, true_mask):
    intersection = np.logical_and(predicted_mask, true_mask)
    union = np.logical_or(predicted_mask, true_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

# Function to perform Mini-Batch K-means segmentation and generate masks
def mini_batch_kmeans_segmentation(images, num_clusters, batch_size=800, max_iters=500):
    masks = []

    for image in tqdm(images, desc="Performing Mini Batch kmeans segmentation"):
        # Flatten the image
        image_flat = image.reshape(-1, 1)

        # Fit Mini-Batch K-means clustering
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=batch_size, max_iter=max_iters).fit(image_flat)

        # Get cluster labels and reshape to the image shape
        cluster_labels = kmeans.labels_.reshape(image.shape)

        # Create a mask based on cluster labels
        mask = (cluster_labels == 1)  # Assuming the object of interest is in cluster 1

        masks.append(mask)

    return masks

# Load and preprocess the data for training, dev, and test sets
#train_images, train_masks = load_and_preprocess_data(train_data_dir, train_mask_dir, 0)
dev_images, dev_masks = load_and_preprocess_data(dev_data_dir, dev_mask_dir, 0)
#test_images, test_masks = load_and_preprocess_data(test_data_dir, test_mask_dir, 1)

# Perform Mini-Batch K-means segmentation on training, dev, and test sets
#train_pred_masks = mini_batch_kmeans_segmentation(train_images, num_clusters)
dev_pred_masks = mini_batch_kmeans_segmentation(dev_images, num_clusters)
#test_pred_masks = mini_batch_kmeans_segmentation(test_images, num_clusters)

# Calculate IoU on dev set
iou_scores_dev = []
for dev_pred_mask, dev_true_mask in tqdm(zip(dev_pred_masks, dev_masks), desc="Calculating IoU (Dev)"):
    iou = calculate_iou(dev_pred_mask, dev_true_mask)
    iou_scores_dev.append(iou)

average_iou_dev = np.mean(iou_scores_dev)
print(f"Average IoU on dev set: {average_iou_dev}")

# Calculate IoU on training set
#iou_scores_train = []
#for train_pred_mask, train_true_mask in tqdm(zip(train_pred_masks, train_masks), desc="Calculating IoU (Train)"):
#    iou = calculate_iou(train_pred_mask, train_true_mask)
#    iou_scores_train.append(iou)

#average_iou_train = np.mean(iou_scores_train)
#print(f"Average IoU on training set: {average_iou_train}")

# Calculate IoU on test set
#iou_scores_test = []
#for test_pred_mask, test_true_mask in tqdm(zip(test_pred_masks, test_masks), desc="Calculating IoU (Test)"):
#    iou = calculate_iou(test_pred_mask, test_true_mask)
#    iou_scores_test.append(iou)

#average_iou_test = np.mean(iou_scores_test)
#print(f"Average IoU on test set: {average_iou_test}")

# Save the predicted masks for the test set
#for i, test_pred_mask in tqdm(enumerate(test_pred_masks), total=len(test_pred_masks), desc="Saving Test Masks"):
#    filename = f"{file_names[i]}"
#    original_image = cv2.imread(os.path.join(test_data_dir, filename), cv2.IMREAD_GRAYSCALE)  # Read as grayscale
#    correct_mask = cv2.imread(os.path.join(test_mask_dir, filename), cv2.IMREAD_GRAYSCALE)  # Read as grayscale

    # Create a subplot image with original image, correct mask, and segmented cell mask
#    plt.figure(figsize=(12, 4))  # Create a larger figure to accommodate subplots

    # Original Image
#    plt.subplot(1, 3, 1)
#    plt.imshow(original_image, cmap='gray')
#    plt.title("Original Image")

    # Correct Mask
#    plt.subplot(1, 3, 2)
#    plt.imshow(correct_mask, cmap='gray')
#    plt.title("Correct Mask")

    # Segmented Cell Mask
#    plt.subplot(1, 3, 3)
#    plt.imshow(original_image, cmap='gray')
#    plt.imshow(test_pred_mask, alpha=0.5, cmap='viridis')
#    plt.title("Segmented Cells\nIoU: {:.4f}".format(iou_scores_test[i]))  # Display IoU score

#    plt.tight_layout()  # Ensure subplots don't overlap

    # Save the subplot image with the same name as the image file
#    subplot_image_path = os.path.join(output_dir, f'{filename}_subplot.png')
#    plt.savefig(subplot_image_path)
#    plt.close()  # Close the figure after saving

    # Save the test mask with the same name as the image file
#    cv2.imwrite(os.path.join(output_dir, filename), (test_pred_mask * 255).astype(np.uint8))