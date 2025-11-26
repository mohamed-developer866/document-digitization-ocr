# Install required packages
!apt-get install -y tesseract-ocr
!pip install pytesseract opencv-python pillow matplotlib numpy

# Import libraries
import pytesseract
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files
import io

# Set Tesseract path for Colab
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Upload image
print("Please upload your image file:")
uploaded = files.upload()

# Get the uploaded filename
uploaded_filename = list(uploaded.keys())[0]
print(f"Uploaded file: {uploaded_filename}")

def preprocess_image(image_path):
    """
    Advanced image preprocessing for OCR
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply different preprocessing techniques and compare results
    processed_images = {}
    
    # 1. Original grayscale
    processed_images['original'] = gray
    
    # 2. Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    processed_images['gaussian_blur'] = blur
    
    # 3. Median blur to reduce noise while preserving edges
    median = cv2.medianBlur(gray, 3)
    processed_images['median_blur'] = median
    
    # 4. Bilateral filter for noise reduction while keeping edges sharp
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    processed_images['bilateral'] = bilateral
    
    # 5. Thresholding - Otsu's method
    _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images['otsu_threshold'] = thresh_otsu
    
    # 6. Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
    processed_images['adaptive_threshold'] = adaptive_thresh
    
    # 7. Morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel)
    processed_images['morphological'] = opening
    
    # 8. Noise removal with additional filtering
    denoised = cv2.fastNlMeansDenoising(adaptive_thresh, None, 30, 7, 21)
    processed_images['denoised'] = denoised
    
    return processed_images, img

def perform_ocr(image, method_name, config=''):
    """
    Perform OCR with different configurations
    """
    results = {}
    
    # Basic configuration
    basic_config = r'--oem 3 --psm 6'
    results['basic'] = pytesseract.image_to_string(image, config=basic_config)
    
    # Configuration for single column text
    single_col_config = r'--oem 3 --psm 4'
    results['single_column'] = pytesseract.image_to_string(image, config=single_col_config)
    
    # Configuration for single uniform text block
    uniform_block_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?()[]{}:; '
    results['uniform_block'] = pytesseract.image_to_string(image, config=uniform_block_config)
    
    # Try with different page segmentation modes
    psm_configs = {
        'psm_3': r'--oem 3 --psm 3',  # Fully automatic page segmentation, but no OSD
        'psm_4': r'--oem 3 --psm 4',  # Assume a single column of text of variable sizes
        'psm_6': r'--oem 3 --psm 6',  # Assume a single uniform block of text
        'psm_8': r'--oem 3 --psm 8',  # Treat the image as a single word
        'psm_13': r'--oem 3 --psm 13' # Raw line. Treat the image as a single text line
    }
    
    for psm_name, psm_config in psm_configs.items():
        results[psm_name] = pytesseract.image_to_string(image, config=psm_config)
    
    return results

def evaluate_text_quality(text):
    """
    Simple evaluation of text quality
    """
    if not text.strip():
        return 0, "Empty"
    
    # Count alphabetic characters vs total characters
    alpha_chars = sum(c.isalpha() for c in text)
    total_chars = len(text.replace('\n', '').replace(' ', ''))
    
    if total_chars == 0:
        return 0, "No characters"
    
    alpha_ratio = alpha_chars / total_chars
    
    # Simple heuristic for quality assessment
    if alpha_ratio > 0.7 and len(text) > 20:
        quality = "High"
        score = 1.0
    elif alpha_ratio > 0.5 and len(text) > 10:
        quality = "Medium"
        score = 0.7
    elif alpha_ratio > 0.3:
        quality = "Low"
        score = 0.4
    else:
        quality = "Poor"
        score = 0.1
    
    return score, quality

def display_results(processed_images, original_image):
    """
    Display all preprocessing results and their OCR outputs
    """
    best_result = ""
    best_score = 0
    best_method = ""
    best_config = ""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    
    methods = list(processed_images.keys())
    
    for idx, method in enumerate(methods):
        if idx >= 9:  # Only show first 9 methods
            break
            
        img_processed = processed_images[method]
        axes[idx].imshow(img_processed, cmap='gray')
        axes[idx].set_title(f'Method: {method}')
        axes[idx].axis('off')
        
        # Perform OCR with different configurations
        ocr_results = perform_ocr(img_processed, method)
        
        # Find the best result for this method
        for config_name, text in ocr_results.items():
            score, quality = evaluate_text_quality(text)
            
            if score > best_score and len(text.strip()) > 0:
                best_score = score
                best_result = text
                best_method = method
                best_config = config_name
            
            print(f"--- {method} - {config_name} (Quality: {quality}) ---")
            print(f"Score: {score:.2f}")
            print(f"Text length: {len(text)}")
            print("First 100 chars:" if len(text) > 100 else "Text:")
            print(text[:100] + "..." if len(text) > 100 else text)
            print("\n")
    
    plt.tight_layout()
    plt.show()
    
    return best_result, best_method, best_config

# Main execution
try:
    # Preprocess the image with multiple techniques
    processed_images, original_img = preprocess_image(uploaded_filename)
    
    print("=" * 80)
    print("ADVANCED OCR PROCESSING")
    print("=" * 80)
    print(f"Processing image: {uploaded_filename}")
    print(f"Image shape: {original_img.shape}")
    print("=" * 80)
    
    # Display results and get the best one
    best_text, best_method, best_config = display_results(processed_images, original_img)
    
    print("=" * 80)
    print("BEST RESULT SUMMARY")
    print("=" * 80)
    print(f"Best preprocessing method: {best_method}")
    print(f"Best OCR configuration: {best_config}")
    print("Extracted Text:")
    print("=" * 80)
    print(best_text)
    print("=" * 80)
    
    # Save the best result to a text file
    output_filename = uploaded_filename.split('.')[0] + '_extracted_text.txt'
    with open(output_filename, 'w') as f:
        f.write(best_text)
    
    print(f"Best result saved to: {output_filename}")
    
    # Offer download
    files.download(output_filename)
    
except Exception as e:
    print(f"Error processing image: {str(e)}")
    print("Please check if the image file was uploaded correctly.")
