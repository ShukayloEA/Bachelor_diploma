from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import torch
import sys

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nanotrack.core.config import cfg
from nanotrack.models.model_builder import ModelBuilder
from nanotrack.tracker.tracker_builder import build_tracker
from nanotrack.utils.model_load import load_pretrain

parser = argparse.ArgumentParser(description='nanotrack') 

parser.add_argument('--tracker_name', '-t', default='nanotrackv3',type=str,help='tracker name')

parser.add_argument('--config', default='./NanoTrack/models/config/configv3.yaml',  type=str,help='config file')

parser.add_argument('--snapshot', default='./NanoTrack/models/pretrained/nanotrackv3.pth', type=str,help='snapshot of models to eval')

parser.add_argument('--save_path', default='./NanoTrack/models/nanotrack_full.onnx', type=str, help='path to ONNX model')

parser.set_defaults(show_video_level=False)

args = parser.parse_args() 

cfg.merge_from_file(args.config) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Создаём модель
model = ModelBuilder()
model = load_pretrain(model, args.snapshot)
model = model.to(device)
model.eval()

# build tracker 
#tracker = build_tracker(model)

#Создаём "пустой" вход
template = torch.randn(1, 3, 127, 127)
search = torch.randn(1, 3, 255, 255)

#model.template(dummy_template)
#model.zf = model.zf.detach()
model.forward = model.forward_onnx  # заменяем forward для ONNX экспорта

#Экспортируем модель
torch.onnx.export(model,
                  (template, search),
                  args.save_path,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['template', 'search'],
                  output_names=['cls', 'loc']
                  )