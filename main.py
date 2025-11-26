import cv2
import pytesseract
import pandas as pd
import os
from datetime import datetime
from PIL import Image

# Path to Tesseract (change ONLY if using Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Create output folders
os.makedirs("outputs/text_files", exist_ok=True)
os.makedirs("outputs/original_files", exist_ok=True)

# Input file (sample)
INPUT_FILE = "sample_document.jpg"    # Add your document here

# Save original copy to folder
original_path = os.path.join("outputs/original_files", INPUT_FILE)
if not os.path.exists(original_path):
    os.system(f"copy {INPUT_FILE} outputs/original_files")

# Read image
img = cv2.imread(INPUT_FILE)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# OCR Extraction
extracted_text = pytesseract.image_to_string(gray)

# Save text
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
text_filename = f"text_{timestamp}.txt"
text_path = os.path.join("outputs/text_files", text_filename)

with open(text_path, "w", encoding="utf-8") as f:
    f.write(extracted_text)

# Excel Index
index_row = {
    "Document Name": INPUT_FILE,
    "Saved Text File": text_filename,
    "Extracted Text Preview": extracted_text[:100],
    "Timestamp": datetime.now()
}

excel_path = "document_index.xlsx"

if os.path.exists(excel_path):
    old_df = pd.read_excel(excel_path)
    new_df = pd.concat([old_df, pd.DataFrame([index_row])], ignore_index=True)
else:
    new_df = pd.DataFrame([index_row])

new_df.to_excel(excel_path, index=False)

print("OCR Completed Successfully!")
print("Text saved to:", text_path)
print("Index updated: document_index.xlsx")
