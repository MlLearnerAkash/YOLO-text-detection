import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.decomposition import PCA
from math import log10

def load_images_from_directory(directory_path):
    images = []
    image_filenames = []
    for filename in os.listdir(directory_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(directory_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale for simplicity
            img = cv2.resize(img, (1024, 1024))
            images.append(img)
            image_filenames.append(filename)
    return images, image_filenames

def calculate_ssim(img1, img2):
    score, _ = ssim(img1, img2, full=True)
    return score

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:  # Avoid division by zero
        return float('inf')
    max_pixel = 255.0
    psnr_value = 10 * log10(max_pixel**2 / mse)
    return psnr_value

def calculate_fft(img):
    fft_result = np.fft.fft2(img)
    fft_magnitude = np.abs(fft_result)
    return fft_magnitude

def calculate_pca(images):
    reshaped_images = [img.flatten() for img in images]
    reshaped_images = np.array(reshaped_images)
    
    pca = PCA(n_components=2)  # Reduce to 2 components for comparison
    pca.fit(reshaped_images)
    transformed_images = pca.transform(reshaped_images)
    return transformed_images

def compare_images(directory_path):
    images, filenames = load_images_from_directory(directory_path)
    n = len(images)
    
    print(f"Comparing {n} images from the directory '{directory_path}'")
    
    # SSIM and PSNR between pairs
    for i in range(n):
        for j in range(i+1, n):
            img1, img2 = images[i], images[j]
            ssim_value = calculate_ssim(img1, img2)
            psnr_value = calculate_psnr(img1, img2)
            print(f"SSIM between {filenames[i]} and {filenames[j]}: {ssim_value:.4f}")
            print(f"PSNR between {filenames[i]} and {filenames[j]}: {psnr_value:.2f} dB")
    
    # FFT for each image
    print("\nFFT magnitudes for each image:")
    for i, img in enumerate(images):
        fft_magnitude = calculate_fft(img)
        print(f"FFT magnitude (mean) for {filenames[i]}: {np.mean(fft_magnitude):.4f}")
    
    # PCA across all images
    pca_result = calculate_pca(images)
    print("\nPCA components (2D projection):")
    for i in range(n):
        print(f"{filenames[i]}: PCA1={pca_result[i, 0]:.4f}, PCA2={pca_result[i, 1]:.4f}")

# Usage
directory_path = '/home/akash/ws/dataset/hand_written/test_data/oriya/val_images'  # Replace with your image directory path
compare_images(directory_path)
