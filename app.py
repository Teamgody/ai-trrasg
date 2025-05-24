from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import tensorflow as tf

# โหลด TFLite model
interpreter = tf.lite.Interpreter(model_path="garbage_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

bin_mapping = {
    'cardboard': ('ขยะรีไซเคิล', 'static/bins/bin_blue.png'),
    'plastic': ('ขยะรีไซเคิล', 'static/bins/bin_blue.png'),
    'metal': ('ขยะอันตราย', 'static/bins/bin_yellow.png'),
    'glass': ('ขยะอันตราย', 'static/bins/bin_yellow.png'),
    'paper': ('ขยะอันตราย', 'static/bins/bin_yellow.png'),  # เปลี่ยนตามคำขอ
    'trash': ('ขยะทั่วไป', 'static/bins/bin_red.png'),
}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB').resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        class_index = np.argmax(output_data)
        class_label = class_names[class_index]

        if class_label in bin_mapping:
            label_thai, bin_img = bin_mapping[class_label]
        else:
            label_thai, bin_img = 'ขยะอินทรีย์', 'static/bins/bin_green.png'

        return label_thai, bin_img  # ✅ เปลี่ยนตรงนี้
    except Exception as e:
        print("Error in classify_image:", e)
        return "ไม่สามารถวิเคราะห์ภาพได้", None

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    bin_image = None

    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result, bin_image = classify_image(filepath)
        else:
            result = "กรุณาอัปโหลดไฟล์รูปภาพที่ถูกต้อง"
            bin_image = None

    return render_template('index.html', result=result, bin_image=bin_image)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
