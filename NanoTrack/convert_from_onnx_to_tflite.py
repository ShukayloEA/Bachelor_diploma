from onnx_tf.backend import prepare
import onnx
import tensorflow as tf

save_path = "./NanoTrack/models/nanotrack_model_tf"

onnx_model = onnx.load('./NanoTrack/models/nanotrack_full.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph(save_path)


converter = tf.lite.TFLiteConverter.from_saved_model(save_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("./NanoTrack/models/nanotrack_model.tflite", "wb") as f:
    f.write(tflite_model)