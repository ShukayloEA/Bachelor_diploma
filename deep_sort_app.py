# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np
import torch

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from opts import opt    

import time
#print(sys.path)
from yolov5.models.common import DetectMultiBackend

import torchvision.models as models

from openvino.runtime import Core
ie = Core()

# Load YOLO models
weights_path_rgb = "exp22\\weights\\best_openvino_model\\best.xml"
#model_rgb = torch.load("C:\\Diploma\\yolov5\\runs\\train\\exp20\\weights\\best.pt", map_location="cpu")
#model_rgb = torch.load(weights_path_rgb, map_location="cpu", weights_only=False)
#weights_path_ir = "C:\\Diploma\\yolov5\\runs\\train\\exp18\\weights\\best.pt"
weights_path_ir = "exp18\\weights\\best_openvino_model\\best.xml"
#model_ir = torch.load("C:\\Diploma\\yolov5\\runs\\train\\exp8\\weights\\best.pt", map_location="cpu")
#model_ir = torch.load(weights_path_ir, map_location="cpu", weights_only=False)
model_ir = ie.compile_model(weights_path_ir, "CPU")
model_rgb = ie.compile_model(weights_path_rgb, "CPU")
device = 'cpu'
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model_rgb = DetectMultiBackend(weights_path_rgb, device=device)
#model_ir = DetectMultiBackend(weights_path_ir, device=device)
#model_rgb = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path_rgb)
#model_ir = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path_ir)
#model_rgb = yolov5.load(weights_path_rgb) 
#model_ir = yolov5.load(weights_path_ir)

#model_rgb.eval()
#model_ir.eval()

def scale_bbox(bbox, orig_shape, img_shape=(640, 640)):
    """ Преобразует bbox из YOLO-координат обратно в исходные размеры изображения. """
    img_w, img_h = img_shape
    orig_w, orig_h = orig_shape[::-1]
    
    # Масштабирование обратно в оригинальный размер
    scale_x = orig_w / img_w
    scale_y = orig_h / img_h

    # Восстанавливаем координаты
    x1_orig = bbox[0] * scale_x
    y1_orig = bbox[1] * scale_y
    w_orig = bbox[2] * scale_x
    h_orig = bbox[3] * scale_y

    #From YOLO format to tlwh
    bbox[0] = x1_orig - w_orig/2
    bbox[1] = y1_orig - h_orig/2
    bbox[2] = w_orig
    bbox[3] = h_orig

    return (bbox[0], bbox[1], bbox[2], bbox[3])

def gather_sequence_info(sequence_dir, is_infrared):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    is_infrared : bool
        If true, use infrared YOLO model, else RGB YOLO model.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    video_type = "infrared" if is_infrared else "visible"
    #image_dir = os.path.join(sequence_dir, video_type)
    image_dir = sequence_dir
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = 1 #int(detections[:, 0].min())
        max_frame_idx = 1 #int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, f"seqinfo_{video_type}.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["framerate"])
    else:
        update_ms = None

    #feature_dim = detections.shape[1] - 10 if detections is not None else 0
    #sequence_name = os.path.join(sequence_dir, video_type)
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "groundtruth": groundtruth,
        "is_infrared": is_infrared,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        #"feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(model, image, min_height=0, min_confidence=0.5):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    '''
    frame_indices = detection_mat[:, 0].astype(np.int64)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list
    '''

    # Model selection
    #model = model_ir if is_infrared else model_rgb
    image = cv2.resize(image, (640, 640))
    # Convert to tensor and normalize
    image = torch.tensor(image, dtype=torch.float32) / 255.0

    #image = torch.tensor(image, dtype=torch.float64).permute(2, 0, 1).unsqueeze(0).to("cpu")
    #image = torch.tensor(image, dtype=torch.float64)
    # If image is infrared (H, W), add chanel (C=1)
    if image.ndim == 2:  
        image = image.unsqueeze(0)  # (H, W) → (1, H, W)
        image = image.repeat(3, 1, 1)  # (1, H, W) → (3, H, W)
    elif image.shape[0] == 1:  
        image = image.repeat(3, 1, 1)  # (1, H, W) → (3, H, W)
    else:  
        image = image.permute(2, 0, 1)  # (H, W, C) → (C, H, W)
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    
    #start = time.time()
    # Get predictions
    with torch.no_grad():
        #print(f"Model confidence threshold: {model.model.conf}")  # Должно вывести min_confidence
        
        #model.model.conf = min_confidence
        #model.model.iou = 0.9
        #start = time.time()
        outputs = model(image)
        #end = time.time() - start
        #print(end)
        
        #print(f"Model confidence threshold: {model.model.conf}")  # Должно вывести min_confidence
        #print(f"Output: {outputs}")
        #print(f"Output shape: {outputs[0].shape}")
    
    # Обработка выходов модели и создание объектов Detection
    #threshold = 0.3  # Порог уверенности
    detections = []
    
    output = outputs[0]
    #for output in outputs:
        #output = output[0]
    #print(f"shape: {output.shape}")
    bbox = output[0, :, :4] # in YOLO format
    confidence = output[0, :, 4]
    feature = output[0, :, 5:]
    
    #start = time.time()
    for i in range(0, len(confidence)):
        if bbox[i, 3] >= min_height and confidence[i] >= min_confidence:  # проверка по высоте bbox
            #bbox[0] = bbox[0] - (bbox[2] / 2)
            #bbox[1] = bbox[1] - (bbox[3] / 2)
            detections.append(Detection(bbox[i], confidence[i], feature[i]))
    #end = time.time() - start
    #print(end)
    #end = time.time() - start
    #print(end)
    return detections



def run(sequence_dir, is_infrared, is_video, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    is_infrared : bool
        If true, use infrared YOLO model, else RGB YOLO model.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine',
        max_cosine_distance,
        nn_budget
    )
    tracker = Tracker(metric)
    results = []
    
    if not is_video:
        seq_info = gather_sequence_info(sequence_dir, is_infrared)
    else:
        cap = cv2.VideoCapture(sequence_dir)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frame_idx = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        
        #out = cv2.VideoWriter(output_video, fourcc, fps, (width, height), isColor=False)
        
        seq_info = {
            "sequence_name": os.path.basename(sequence_dir),
            "image_filenames": None,
            "groundtruth": None,
            "is_infrared": is_infrared,
            "image_size": (height, width),
            "min_frame_idx": 0,
            "max_frame_idx": max_frame_idx,
            "update_ms": 1000 / fps
        }
    
    model = model_ir if is_infrared else model_rgb

    def frame_callback(vis, frame_idx):
        # Load image and generate detections.
        if not is_video:
            # Get filepath
            #video_type = "infrared" if is_infrared else "visible"
            image_path = os.path.join(sequence_dir, f"{frame_idx:06d}.jpg")
            if not os.path.exists(image_path):
                print(f"File {image_path} is not found!")
                return []

            # Load and image preprocessing
            image = cv2.imread(image_path, cv2.IMREAD_COLOR if not is_infrared else cv2.IMREAD_GRAYSCALE)
            
            detections = create_detections(
                model, image, min_detection_height, min_confidence)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                detections = create_detections(
                model, frame, min_detection_height, min_confidence)

        #detections = [d for d in detections if d.confidence >= min_confidence]

        #for d in range(detections):
        #    d.tlwh = scale_bbox(d.tlwh, seq_info["image_size"][0], seq_info["image_size"][0])
        # Масштабируем bbox обратно в оригинальные размеры
        #detections = [Detection(scale_bbox(*det.tlwh, seq_info["image_size"][0], seq_info["image_size"][1]), det.confidence, det.feature) for det in detections]
        # Run non-maxima suppression.
        boxes = np.array([scale_bbox(d.tlwh, seq_info["image_size"]) for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            if not is_video:
                image = cv2.imread(
                    seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            else:
                image = frame
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)
        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                    frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
            
        #if is_video:
        #    vis.store(out)

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=40)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    start = time.time()
    visualizer.run(frame_callback)
    end = time.time() - start
    print(end)
    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
    #print('%d' % (seq_info["max_frame_idx"]),file=f)
    f.close()
    if is_video:
        cap.release()
        #out.release()

def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)
