import torch
import cv2
import os
import easyocr
import numpy as np

# inisialisasi model default
model = torch.hub.load('ultralytics/yolov5', 'custom', 'models/best.onnx')
model.conf = 0.80

# Initialize EasyOCR
reader = easyocr.Reader(['id'])

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

def get_detections():
    directory_path = 'img_plate'
    bottom_folder = get_bottom_folder(directory_path)

    crop_path = os.path.join(bottom_folder, 'crops')

    jpg_files = get_jpg_files(crop_path)
    return jpg_files

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

def drawings(image,crop_dir):
    # drawings
    text_list = []
    for file_crop in crop_dir:
        license_text = ocr(file_crop)
        text_list.append(license_text)

    # Overlay the plate number on the original image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 5
    font_color = (0, 255, 0)  # BGR format (blue, green, red)
    text_position = (100, 250)  # Adjust the position as needed

    cv2.putText(image, license_text, text_position, font, font_scale, font_color, font_thickness)

    return image, text_list, crop_dir[0]    

def yolo_predictions(img,model):
    ## step-1: detections
    results = model(img)
    results.crop(save=True, save_dir='img_plate/exp') 
    crop_dir = get_detections()
    ## step-2: Drawings
    result_img, text, crop_img = drawings(img,crop_dir)
    return result_img, text, crop_img


def object_detection_default(path,filename):
    # read image
    image = cv2.imread(path) # PIL object

    # get the shape
    h,w,d = image.shape
    
    #crop
    # Define the percentage of the image to be cropped from both sides
    crop_percentage = 15 

    # Calculate the number of pixels to crop from both sides
    crop_pixels = int(w * crop_percentage / 100)

    # Crop the image
    img_crop = image[:, crop_pixels:w-crop_pixels]

    # resize
    img_res = cv2.resize(img_crop, (640, 640))

    # grayscale
    img_gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

    # image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    result_img, text_list, crop_img = yolo_predictions(img_gray,model)
    cropped = cv2.imread(crop_img)
    cv2.imwrite('./static/predict/{}'.format(filename),result_img)
    cv2.imwrite('./static/crop/{}'.format(filename),img_crop)
    cv2.imwrite('./static/resize/{}'.format(filename),img_res)
    cv2.imwrite('./static/gray/{}'.format(filename),img_gray)
    cv2.imwrite('./static/roi/{}'.format(filename),cropped)

    return text_list

def object_detection_custom(path,filename,mod_path):
    # read image
    image = cv2.imread(path) # PIL object

    # get the shape
    h,w,d = image.shape
    
    #crop
    # Define the percentage of the image to be cropped from both sides
    crop_percentage = 15 

    # Calculate the number of pixels to crop from both sides
    crop_pixels = int(w * crop_percentage / 100)

    # Crop the image
    img_crop = image[:, crop_pixels:w-crop_pixels]

    # resize
    img_res = cv2.resize(img_crop, (640, 640))

    # grayscale
    img_gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

    model_custom = torch.hub.load('ultralytics/yolov5', 'custom', mod_path)
    model_custom.conf = 0.80

    # image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    result_img, text_list, crop_img = yolo_predictions(img_gray,model_custom)
    cropped = cv2.imread(crop_img)
    cv2.imwrite('./static/predict/{}'.format(filename),result_img)
    cv2.imwrite('./static/crop/{}'.format(filename),img_crop)
    cv2.imwrite('./static/resize/{}'.format(filename),img_res)
    cv2.imwrite('./static/gray/{}'.format(filename),img_gray)
    cv2.imwrite('./static/roi/{}'.format(filename),cropped)

    return text_list