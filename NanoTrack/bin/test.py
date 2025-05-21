# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

import time
import sys 
#sys.path.append(os.getcwd())  
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..\\..')))
from tqdm import tqdm 

#import onnx
#import onnxruntime
#import tensorflow as tf

#from nanotrack.models.onnx_wrapper import ONNXWrapper
from nanotrack.models.tflite_wrapper import TFLiteWrapper
from nanotrack.core.config import cfg
from nanotrack.models.model_builder import ModelBuilder
from nanotrack.tracker.tracker_builder import build_tracker
from nanotrack.utils.bbox import get_axis_aligned_bbox
from nanotrack.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str

from bin.eval import eval

from angular_offset import calculate_angular_offset, calculate_vertical_fov

parser = argparse.ArgumentParser(description='nanotrack') 

#parser.add_argument('--dataset', default='Anti-UAV-RGBT', type=str,help='datasets')

parser.add_argument('--tracker_name', '-t', default='nanotrackv3',type=str,help='tracker name')

parser.add_argument('--config', default='./models/config/configv3.yaml',  type=str,help='config file')

parser.add_argument('--snapshot', default='models/pretrained/nanotrackv3.pth', type=str,help='snapshot of models to eval')

parser.add_argument('--save_path', default='./results', type=str, help='snapshot of models to eval')

parser.add_argument('--video', default='.\\datasets\\20190925_101846_1_1\\visible.mp4', type=str,  help='eval one special video')

parser.add_argument('--fov', default=46, type=float, help='fov of drone camera')

parser.add_argument('--vis', action='store_true',help='whether visualize result')

parser.add_argument('--gpu_id', default='not_set', type=str, help="gpu id") 

parser.add_argument('--tracker_path', '-p', default='./results', type=str,help='tracker result path')

parser.add_argument('--num', '-n', default=4, type=int,help='number of thread to eval')

parser.add_argument('--show_video_level', '-s', dest='show_video_level',action='store_true')


parser.set_defaults(show_video_level=False)

args = parser.parse_args() 

if args.gpu_id != 'not_set': 

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 

torch.set_num_threads(1)  

def main(): 
    '''
    onnx_model_path = "./models/nanotrack_full.onnx"
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    session = ONNXWrapper(onnx_model_path)
    '''
    anno_path = os.path.splitext(args.video)[0] + "_gt.txt"
    annotations = []
    if os.path.exists(anno_path):
        with open(anno_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Пример: "x,y,w,h" или "frame, x, y, w, h"
                    parts = list(map(float, line.split(',')))
                    # Если в начале номера кадра, можно пропускать первый элемент:
                    # bbox = parts[1:]
                    annotations.append(parts)
    # иначе annotations = []

    tflite_model_path="./models/nanotrack_model.tflite"
    tflite_model = TFLiteWrapper(tflite_model_path)

    cfg.merge_from_file(args.config) 

    #dataset_root = os.path.join('./datasets', args.dataset) 
                  
    params = [0.0,0.0,0.0]
    
    params[0] =cfg.TRACK.LR 
    params[1]=cfg.TRACK.PENALTY_K
    params[2] =cfg.TRACK.WINDOW_INFLUENCE 
    '''                                               args.dataset'''
    params_name = args.snapshot.split('/')[-1] + ' ' + os.path.basename(args.video) + '  lr-' + str(params[0]) + '  pk-' + '_' + str(params[1]) + '  win-' + '_' + str(params[2])
    
    # create model 
    #model = ModelBuilder() 

    # load model 
    device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
    #model = load_pretrain(model, args.snapshot).to(device).eval()

    # build tracker 
    #tracker = build_tracker(model)
    tracker = build_tracker(tflite_model)

    # create dataset 
    #dataset = DatasetFactory.create_dataset(name=args.dataset,  
    #                                        dataset_root=dataset_root,
    #                                        load_img=False)  
    

    #for video in tqdm(enumerate(dataset)):
    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start = time.time()
    # if args.video != '':
    #     # test one special video
    #     if video.name != args.video:
    #         continue
    toc = 0
    idx = 0
    pred_bboxes = []
    angular_offsets = []
    scores = []
    track_times = []
    
    fov_h = args.fov #horisontal
    fov_v = calculate_vertical_fov(fov_h, width/height) #vertical
    #for idx, (img, gt_bbox) in enumerate(video):
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        idx += 1

        if len(annotations) > 0 and idx == annotations[idx-1][0]:
            gt_bbox = annotations[idx-1]
        else:
            gt_bbox = [idx, 1, width/2, height/2, 1, 1, 1]

        tic = cv2.getTickCount()
        if idx == 1:
            cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
            gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h] #[topx,topy,w,h]
            tracker.init(img, gt_bbox_)
            pred_bbox = gt_bbox_
            scores.append(None)
            
            pred_bboxes.append(pred_bbox)
        else: 
            outputs = tracker.track(img)
            pred_bbox = outputs['bbox']
            pred_bboxes.append(pred_bbox)
            #scores.append(outputs['best_score'])  
        toc += cv2.getTickCount() - tic
        track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())

        angular_offset = calculate_angular_offset(cx, cy, width, height, fov_h, fov_v)
        angular_offsets.append(angular_offset)

        if idx == 1:
            cv2.destroyAllWindows()
        if args.vis and idx > 0: 
            gt_bbox = list(map(int, gt_bbox))
            pred_bbox = list(map(int, pred_bbox))
            cv2.rectangle(img, (gt_bbox[2], gt_bbox[3]),
                        (gt_bbox[2]+gt_bbox[4], gt_bbox[3]+gt_bbox[5]), (0, 255, 0), 3)
            cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                        (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
            cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow(os.path.basename(args.video), img)
            cv2.waitKey(1)
        
        # Прерывание по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    toc /= cv2.getTickFrequency()
    cap.release()
    cv2.destroyAllWindows()
    end = time.time() - start
    print(end)
    # save results 

    model_path = os.path.join(args.save_path, args.dataset, args.tracker_name)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    result_path = os.path.join(model_path, '{}.txt'.format(video.name))
    with open(result_path, 'w') as f:
        frame = 1
        for x in pred_bboxes:
            f.write(str(frame))
            f.write(',1,')
            frame += 1
            f.write(','.join([str(i) for i in x]))   
            f.write(',1\n')
    
    result_path = os.path.join(model_path, '{}_angular_offsets.txt'.format(video.name))
    with open(result_path, 'w') as f:
        frame = 1
        for x in angular_offsets:
            f.write(str(frame))
            f.write(',')
            frame += 1
            f.write(','.join([str(i) for i in x]))   
            f.write('\n')


if __name__ == '__main__':
    main()
