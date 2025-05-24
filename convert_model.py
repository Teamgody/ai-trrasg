import tensorflow as tf

# โหลดโมเดล Keras (.h5)
model = tf.keras.models.load_model('garbage_model.h5')

# แปลงโมเดลเป็น TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# บันทึกไฟล์ .tflite
with open('garbage_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("แปลงโมเดลเสร็จแล้ว! ไฟล์ garbage_model.tflite ถูกสร้าง")
