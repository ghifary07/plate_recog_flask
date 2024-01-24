from roboflow import Roboflow
import torch
import cv2
import os
import easyocr
import re

# inisialisasi model
model = torch.hub.load('ultralytics/yolov5', 'custom', 'models/best.onnx')
model.conf = 0.80

# Initialize EasyOCR with the desired language(s)
reader = easyocr.Reader(['id'])

# read image
image = cv2.imread('Dataset/Dataset Plat Nomor/K1/K1_AA5767EY.jpg') #PIL Object
# image = np.array(image, dtype=np.uint8) # 8 bit array (0,255)
# image1 = load_img(path, target_size=(224,224))

# data preprocessing
# image_arr_224 = img_to_array(image1)/255.0 # convert into array and normalize output

# get the shape
h,w,d = image.shape

#crop
# Define the percentage of the image to be cropped from both sides
crop_percentage = 15  # You can adjust this value based on your needs

# Calculate the number of pixels to crop from both sides
crop_pixels = int(w * crop_percentage / 100)

# Crop the image
img_crop = image[:, crop_pixels:w-crop_pixels]

# resize
img_res = cv2.resize(img_crop, (640, 640))

# grayscale
img_gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY) 

# prediction
results = model(img_gray)
roi = results.crop(save=True, save_dir='img_plate/exp') 
print(roi)

# Char recog
def get_bottom_folder(directory):
    # Get all the subdirectories in the given directory
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    # If there are no subdirectories, the current directory is the bottom folder
    if not subdirectories:
        return directory

    # Filter out subdirectories that match the expected naming pattern (start with 'exp')
    exp_folders = [sub for sub in subdirectories if sub.startswith('exp')]

    # If there are no matching subdirectories, the current directory is the bottom folder
    if not exp_folders:
        return directory

    # Extract numerical values from the folder names, or use 0 if no numerical part is found
    folder_values = [int(folder[3:]) if folder[3:].isdigit() else 0 for folder in exp_folders]

    # Find the index of the folder with the highest numerical value
    max_index = folder_values.index(max(folder_values))

    # Recursively find the bottom folder in the selected subdirectory
    return get_bottom_folder(os.path.join(directory, exp_folders[max_index]))

def get_jpg_files(directory):
    # Get a list of all '.jpg' files in the given directory and its subdirectories
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', 'jpeg', '.webp')):
                image_files.append(os.path.join(root, file))
    return image_files

def ocr(roi_dir):
    result = reader.readtext(roi_dir)
    # Extract the recognized text (plate number) from the OCR result
    if result:
        plate_number = []
        exclude = [".", ",", "'", "%", ""]
        for i in range(min(3, len(result))):
                if str(result[i][1]) != exclude:
                    plate_number.append(str(result[i][1])) # Assuming the plate number is in the first result
            
        # Join all plate numbers into a single string
        merged_plate_numbers = ' '.join(plate_number)

    return merged_plate_numbers


# Example usage
directory_path = 'img_plate'
bottom_folder = get_bottom_folder(directory_path)
print(f"The very bottom 'exp' folder in {directory_path} is: {bottom_folder}")

crop_path = os.path.join(bottom_folder, 'crops')

jpg_files = get_jpg_files(crop_path)

print(f"All image files inside the deepest 'exp' folder ({crop_path}):")
# OCR
for jpg_file in jpg_files:
    plate_num = ocr(jpg_file)
    print(plate_num)
   
