import onnxruntime as ort
import numpy as np
import torch

class ONNXWrapper:
    def __init__(self, onnx_model_path):
        self.session = ort.InferenceSession(onnx_model_path)
        self.zf = None  # шаблон будет сохраняться здесь

    def template(self, template_tensor):
        self.zf = template_tensor.cpu().numpy().astype(np.float32)

    def track(self, search_tensor):
        if self.zf is None:
            raise ValueError("Template not set. Call template() before track().")

        search_np = search_tensor.cpu().numpy().astype(np.float32)

        inputs = {
            "template": self.zf,
            "search": search_np
        }

        # Запускаем инференс
        outputs = self.session.run(None, inputs)
        cls, loc = outputs

        cls = torch.from_numpy(cls)
        loc = torch.from_numpy(loc)
        return {'cls': cls, 'loc': loc}
