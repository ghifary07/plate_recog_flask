from flask import Flask, render_template, request, send_file
import os
import zipfile
from single_detect import object_detection_default, object_detection_custom
from batch_detect import batch_detection_default, batch_detection_custom

# webserver gateway interface
app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')
MODEL_PATH = os.path.join(BASE_PATH,'static/model_custom/')
BATCH_PATH = os.path.join(BASE_PATH,'static/batch/')

def create_unique_folder():
    # Get the current count of existing batches
    existing_batches = [folder for folder in os.listdir(BATCH_PATH) if folder.startswith('batch')]
    batch_number = len(existing_batches) + 1

    folder_name = f'batch{batch_number}'
    return folder_name

@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        # Check if the form submitted is for uploading a model
        if 'image_name' and 'model_custom' in request.files:
            uploaded_mod = request.files['model_custom']

            # Save the uploaded model to a temporary location
            filename = uploaded_mod.filename
            path_mod_save = os.path.join(MODEL_PATH,filename)
            uploaded_mod.save(path_mod_save)

            # Load the user-provided model
            user_model = path_mod_save

            upload_file = request.files['image_name']
            filename = upload_file.filename
            path_save = os.path.join(UPLOAD_PATH,filename)
            upload_file.save(path_save)
            text = object_detection_custom(path_save, filename, user_model)

            return render_template("index.html", upload=True, upload_image=filename, text=text)
        
        elif 'image_name' in request.files:
            upload_file = request.files['image_name']
            filename = upload_file.filename
            path_save = os.path.join(UPLOAD_PATH,filename)
            upload_file.save(path_save)
            text = object_detection_default(path_save, filename)

            return render_template("index.html", upload=True, upload_image=filename, text=text)

    return render_template('index.html',upload=False)

@app.route('/batch',methods=['POST','GET'])
def batch():
    global txt_file_path
    if request.method == 'POST':
        # Check if the form submitted is for uploading a model
        if 'image_batch' and 'model_custom' in request.files:
            uploaded_mod = request.files['model_custom']

            # Save the uploaded model to a temporary location
            filename = uploaded_mod.filename
            path_mod_save = os.path.join(MODEL_PATH,filename)
            uploaded_mod.save(path_mod_save)

            # Load the user-provided model
            user_model = path_mod_save

            upload_file = request.files['image_batch']
            filename = upload_file.filename
            # Generate a unique folder name
            folder_name = create_unique_folder()
            unzip_path = os.path.join(BATCH_PATH, folder_name)

            # Save the zip file
            zip_path = os.path.join(BATCH_PATH, filename)
            upload_file.save(zip_path)

            # Unzip the file to the unique folder
            os.makedirs(unzip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            # # Get a list of image files in the unzipped folder
            # image_files = [filename for filename in os.listdir(unzip_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

            txt_file_path = batch_detection_custom(unzip_path, user_model)

            return render_template("batch.html", upload=True, txt_file_path=txt_file_path)
        
        elif 'image_batch' in request.files:
            upload_file = request.files['image_batch']
            filename = upload_file.filename
            # Generate a unique folder name
            folder_name = create_unique_folder()
            unzip_path = os.path.join(BATCH_PATH, folder_name)

            # Save the zip file
            zip_path = os.path.join(BATCH_PATH, filename)
            upload_file.save(zip_path)

            # Unzip the file to the unique folder
            os.makedirs(unzip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            # # Get a list of image files in the unzipped folder
            # image_files = [filename for filename in os.listdir(unzip_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

            txt_file_path = batch_detection_default(unzip_path)

            return render_template("batch.html", upload=True, txt_file_path=txt_file_path)

    return render_template('batch.html',upload=False)

@app.route('/batch_download')
def batch_download():
    return send_file(txt_file_path, as_attachment=True)

@app.route('/training')
def training():
    return render_template("training.html")


if __name__ =="__main__":
    app.run(debug=True)