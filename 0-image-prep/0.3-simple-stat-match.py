import os
import cv2
import numpy as np

def adjust_channel(channel, target_mean, target_std):
    """Adjust a single channel based on target mean and standard deviation."""
    channel_float = channel.astype(np.float32)
    
    # Calculate current mean and std deviation
    mean = np.mean(channel_float)
    std = np.std(channel_float)
    
    # Normalize the channel to 0 mean and 1 std deviation
    normalized = (channel_float - mean) / (std + 1e-7)
    
    # Scale and shift to desired mean and std deviation
    adjusted = normalized * target_std + target_mean
    return adjusted.clip(0, 255).astype(np.uint8)

def adjust_color_image(img, target_means, target_stds):
    """Adjust a color image based on target means and standard deviations for each channel."""
    channels = cv2.split(img)
    adjusted_channels = [adjust_channel(ch, mean, std) for ch, mean, std in zip(channels, target_means, target_stds)]
    return cv2.merge(adjusted_channels)

def process_batch(src_directory, dest_directory, target_means, target_stds):
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    for subdir, dirs, files in os.walk(src_directory):
        for file in files:
            if "wash" not in file and file.endswith(".bmp"):
                src_path = os.path.join(subdir, file)
                dest_path = os.path.join(dest_directory, file)
                
                # Read the image
                image = cv2.imread(src_path)
                
                # Adjust the image
                adjusted_image = adjust_color_image(image, target_means, target_stds)
                
                # Save the adjusted image
                cv2.imwrite(dest_path, adjusted_image)
                print(f"Processed: {file}")

# Define desired means and std deviations for each channel (can be adjusted or computed from the dataset)
target_means = [127.5, 127.5, 127.5]  # midpoint of 0-255 for R, G, and B
target_stds = [50, 50, 50]  # arbitrary values for R, G, and B

src_dir = "/home/data/refined/candescence/tlv/0.2-images_cut/all-final"
dest_dir = "/home/data/refined/candescence/tlv/0.2-images_cut/all_simple_statmatch"
process_batch(src_dir, dest_dir, target_means, target_stds)