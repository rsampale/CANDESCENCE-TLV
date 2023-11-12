import os
import cv2
import numpy as np

# THIS FILE ONLY DOES BATCH HISTOGRAM NORMALIZATION ON NON-WASH or WASH COLONIES SEPARATELY
# IT SAVES WHICHEVER TYPE TO THE DESTINATION DIRECTORY AFTERWARD

def compute_avg_histograms(src_directory):
    histograms_b = []
    histograms_g = []
    histograms_r = []

    for subdir, dirs, files in os.walk(src_directory):
        for file in files:
            if "wash" not in file and file.endswith(".bmp"):
                src_path = os.path.join(subdir, file)
                
                # Read the image
                image = cv2.imread(src_path)
                
                # Split channels
                b, g, r = cv2.split(image)
                
                # Compute histograms for each channel
                hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
                hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
                hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
                
                histograms_b.append(hist_b)
                histograms_g.append(hist_g)
                histograms_r.append(hist_r)

    # Compute the average histograms
    avg_hist_b = sum(histograms_b) / len(histograms_b)
    avg_hist_g = sum(histograms_g) / len(histograms_g)
    avg_hist_r = sum(histograms_r) / len(histograms_r)
    return avg_hist_b, avg_hist_g, avg_hist_r

def histogram_equalization_from_avg_hist(channel, avg_hist):
    cdf_avg = avg_hist.cumsum()
    cdf_avg_normalized = (cdf_avg - cdf_avg.min()) * 255 / (cdf_avg.max() - cdf_avg.min())
    cdf_avg_normalized = cdf_avg_normalized.astype('uint8')
    equalized_channel = cdf_avg_normalized[channel]
    return equalized_channel

def process_images_with_avg_hists(src_directory, dest_directory, avg_hists):
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    for subdir, dirs, files in os.walk(src_directory):
        for file in files:
            if "wash" not in file and file.endswith(".bmp"):
                src_path = os.path.join(subdir, file)
                dest_path = os.path.join(dest_directory, file)
                
                # Read the image
                image = cv2.imread(src_path)
                
                # Split channels
                b, g, r = cv2.split(image)
                
                # Equalize using the average histograms
                b_eq = histogram_equalization_from_avg_hist(b, avg_hists[0])
                g_eq = histogram_equalization_from_avg_hist(g, avg_hists[1])
                r_eq = histogram_equalization_from_avg_hist(r, avg_hists[2])
                
                # Merge equalized channels
                equalized_img = cv2.merge([b_eq, g_eq, r_eq])
                
                # Save the equalized image to the destination directory
                cv2.imwrite(dest_path, equalized_img)
                print(f"Processed: {file}")

src_dir = "/home/data/refined/candescence/tlv/0.2-images_cut/all-final"
dest_dir = "/home/data/refined/candescence/tlv/0.2-images_cut/all-batchnorm"
avg_hists = compute_avg_histograms(src_dir)
process_images_with_avg_hists(src_dir, dest_dir, avg_hists)

# Conclusion: colonies look very odd and display multiple different background colors and intensities