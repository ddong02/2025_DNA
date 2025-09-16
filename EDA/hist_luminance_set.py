import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze_and_plot_dataset_brightness(dataset_dir, bins=50):
    """
    Calculates and prints the average brightness per folder in the dataset,
    and plots a histogram of the overall image brightness distribution.

    Args:
        dataset_dir (str): Path to the top-level directory containing the 'set' folders.
        bins (int): The number of bins for the histogram.
    """
    # A list to store the average brightness value of each individual image.
    all_images_brightness_values = []
    
    # List of subfolders to process, changed to 'set1', 'set2', 'set3'.
    subfolders = ['set1', 'set2', 'set3']
    
    # Supported image extensions.
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    print("="*50)
    print("Starting dataset analysis")
    print("="*50)

    # Iterate through each subfolder.
    for folder_name in subfolders:
        folder_path = os.path.join(dataset_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            print(f"\nWarning: Folder '{folder_path}' not found. Skipping.")
            continue
            
        print(f"\n>>> Processing folder '{folder_name}'...")
        
        folder_pixel_sum = 0
        folder_pixel_count = 0
        
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_extensions)]
        
        for filename in tqdm(image_files, desc=f"Processing {folder_name}"):
            image_path = os.path.join(folder_path, filename)
            try:
                # Load the image in grayscale.
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # 1. Accumulate pixel sum and count for per-folder average calculation.
                    folder_pixel_sum += np.sum(img, dtype=np.int64)
                    folder_pixel_count += img.size
                    
                    # 2. Store the average brightness of the individual image for the overall histogram.
                    all_images_brightness_values.append(np.mean(img))

            except Exception as e:
                tqdm.write(f"Error: A problem occurred while processing '{filename}' - {e}")

        # Calculate and print the average brightness for the folder.
        if folder_pixel_count > 0:
            folder_avg_brightness = folder_pixel_sum / folder_pixel_count
            print(f"✅ Average brightness for folder '{folder_name}': {folder_avg_brightness:.2f}")
        else:
            print(f"❌ No images to process in folder '{folder_name}'.")

    # --- Plotting the histogram for the entire dataset ---
    print("\n" + "="*50)
    print("Generating brightness distribution histogram for the entire dataset...")
    print("="*50)

    if not all_images_brightness_values:
        print("No data available to plot histogram.")
        return

    plt.figure(figsize=(12, 7))
    
    # Create the histogram.
    plt.hist(all_images_brightness_values, bins=bins, color='skyblue', edgecolor='black')
    
    # Set graph title and labels.
    plt.title('Image Brightness Distribution for the Entire Training Dataset (Set1-3)', fontsize=16)
    plt.xlabel('Average Brightness (0=Black, 255=White)', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Display the overall mean as a vertical line.
    overall_mean = np.mean(all_images_brightness_values)
    plt.axvline(overall_mean, color='crimson', linestyle='dashed', linewidth=2)
    plt.text(overall_mean * 1.05, plt.ylim()[1] * 0.9, f'Overall Mean: {overall_mean:.2f}', color='crimson', fontsize=12)

    print(f"📊 Overall average brightness of all images in the dataset: {overall_mean:.2f}")
    
    # Show the plot.
    plt.show()


# --- Main execution block ---
if __name__ == "__main__":
    # ⚠️ Set the path to your top-level training dataset folder here.
    # (The folder containing 'set1', 'set2', 'set3', etc.)
    TRAIN_DATASET_PATH = '/Users/eogus/Desktop/Dataset/2025DNA/SemanticDataset_final/image/train'

    if not os.path.isdir(TRAIN_DATASET_PATH):
        print(f"Error: '{TRAIN_DATASET_PATH}' is not a valid directory. Please check the path.")
    else:
        analyze_and_plot_dataset_brightness(TRAIN_DATASET_PATH)