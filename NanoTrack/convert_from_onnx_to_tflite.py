'''from onnx_tf.backend import prepare
import onnx
import tensorflow as tf

save_path = "C:\\Users\\Aila\\Bachelor_diploma\\exp18\\weights\\best_tf"

onnx_model = onnx.load('C:\\Users\\Aila\\Bachelor_diploma\\exp18\\weights\\best.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph(save_path)


converter = tf.lite.TFLiteConverter.from_saved_model(save_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("C:\\Users\\Aila\\Bachelor_diploma\\exp18\\weights\\best.tflite", "wb") as f:
    f.write(tflite_model)
'''

import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Загрузка ONNX модели
onnx_model = onnx.load('C:\\Users\\Aila\\Bachelor_diploma\\exp18\\weights\\best.onnx')

# Подготовка модели для TensorFlow
tf_rep = prepare(onnx_model)
# Экспорт модели в формат TensorFlow (SavedModel)
tf_rep.export_graph('C:\\Users\\Aila\\Bachelor_diploma\\exp18\\weights\\best_tf')


converter = tf.lite.TFLiteConverter.from_saved_model('C:\\Users\\Aila\\Bachelor_diploma\\exp18\\weights\\best_tf')
tflite_model = converter.convert()

# Сохраняем модель в формате TFLite
with open('C:\\Users\\Aila\\Bachelor_diploma\\exp18\\weights\\best.tflite', 'wb') as f:
    f.write(tflite_model)