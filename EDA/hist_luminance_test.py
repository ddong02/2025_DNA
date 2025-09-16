import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze_single_folder_brightness(dataset_dir, bins=50):
    """
    Calculates the brightness for each image in a single folder and
    plots a histogram of the overall brightness distribution.

    Args:
        dataset_dir (str): Path to the directory containing the image files.
        bins (int): The number of bins for the histogram.
    """
    # A list to store the average brightness value of each individual image.
    all_images_brightness_values = []
    
    # Supported image extensions.
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    print("="*50)
    print(f"Starting analysis for directory: {dataset_dir}")
    print("="*50)

    image_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(supported_extensions)]

    if not image_files:
        print("Error: No images found in the specified directory.")
        return

    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(dataset_dir, filename)
        try:
            # Load the image in grayscale.
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Store the average brightness of the individual image.
                all_images_brightness_values.append(np.mean(img))

        except Exception as e:
            tqdm.write(f"Error: A problem occurred while processing '{filename}' - {e}")

    # --- Plotting the histogram for the dataset ---
    print("\n" + "="*50)
    print("Generating brightness distribution histogram...")
    print("="*50)

    if not all_images_brightness_values:
        print("No data available to plot histogram.")
        return

    plt.figure(figsize=(12, 7))
    
    # Create the histogram.
    plt.hist(all_images_brightness_values, bins=bins, color='mediumseagreen', edgecolor='black')
    
    # Set graph title and labels.
    plt.title('Image Brightness Distribution for the Test Dataset', fontsize=16)
    plt.xlabel('Average Brightness (0=Black, 255=White)', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Display the overall mean as a vertical line.
    overall_mean = np.mean(all_images_brightness_values)
    plt.axvline(overall_mean, color='crimson', linestyle='dashed', linewidth=2)
    plt.text(overall_mean * 1.05, plt.ylim()[1] * 0.9, f'Overall Mean: {overall_mean:.2f}', color='crimson', fontsize=12)

    print(f"📊 Overall average brightness of all images: {overall_mean:.2f}")
    
    # Show the plot.
    plt.show()


# --- Main execution block ---
if __name__ == "__main__":
    # ⚠️ Set the path to your test dataset folder here.
    TEST_DATASET_PATH = '/Users/eogus/Desktop/Dataset/2025DNA/SemanticDatasetTest/image/test/set1'

    if not os.path.isdir(TEST_DATASET_PATH):
        print(f"Error: '{TEST_DATASET_PATH}' is not a valid directory. Please check the path.")
    else:
        analyze_single_folder_brightness(TEST_DATASET_PATH)