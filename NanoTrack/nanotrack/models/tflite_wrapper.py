import numpy as np
import torch
import tensorflow as tf

class TFLiteWrapper:
    def __init__(self, tflite_model_path):
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.template_input_idx = None
        self.search_input_idx = None
        self.cls_output_idx = None
        self.loc_output_idx = None

        self._map_io_indices()
        self.zf = None

        print("Input details:")
        for detail in self.input_details:
            print(f"  name: {detail['name']}, shape: {detail['shape']}")


    def _map_io_indices(self):
        # Предположим: первый вход — шаблон, второй — поиск
        if len(self.input_details) != 2:
            raise RuntimeError("Expected 2 input tensors (template and search), got: {}".format(len(self.input_details)))
        self.template_input_idx = self.input_details[1]['index']
        self.search_input_idx = self.input_details[0]['index']

        if len(self.output_details) != 2:
            raise RuntimeError("Expected 2 output tensors (cls and loc), got: {}".format(len(self.output_details)))
        self.cls_output_idx = self.output_details[0]['index']
        self.loc_output_idx = self.output_details[1]['index']

    def template(self, template_tensor):
        self.zf = template_tensor.cpu().numpy().astype(np.float32)

    def track(self, search_tensor):
        if self.zf is None:
            raise ValueError("Template not set. Call template() before track().")

        search_np = search_tensor.cpu().numpy().astype(np.float32)

        # Задаём входы
        self.interpreter.set_tensor(self.template_input_idx, self.zf)
        self.interpreter.set_tensor(self.search_input_idx, search_np)

        # Запускаем инференс
        self.interpreter.invoke()

        cls = self.interpreter.get_tensor(self.cls_output_idx)
        loc = self.interpreter.get_tensor(self.loc_output_idx)

        return {
            'cls': torch.from_numpy(cls),
            'loc': torch.from_numpy(loc)
        }
