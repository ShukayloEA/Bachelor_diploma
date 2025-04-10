"""
@Author: Du Yunhao
@Filename: strong_sort.py
@Contact: dyh_bupt@163.com
@Time: 2022/2/28 20:14
@Discription: Run StrongSORT
"""


import warnings
from os.path import join, isdir, isfile
warnings.filterwarnings("ignore")
from opts import opt
from deep_sort_app import run

#import time

if __name__ == '__main__':
    #start = time.time()
    #for i, seq in enumerate(opt.sequences, start=1):
    print('processing the video {}...'.format(opt.sequence))
    #path_output_video = join(opt.dir_save, seq + '.mp4')

    if isdir(opt.sequence):
        is_video = False
        path_save = join(opt.sequence + '.txt')
    elif isfile(opt.sequence):
        is_video = True
        path_save = join(opt.sequence[:-4] + '.txt')
    else:
        print("Error: please specify an existing file or folder.")
        exit(1)
    
    run(
        sequence_dir=opt.sequence,#join(opt.dir_dataset, seq),
        is_infrared=opt.is_infrared,
        is_video=is_video,
        #detection_file=join(opt.dir_dets, seq + '.npy'),
        output_file=path_save,
        #output_video=path_output_video,
        min_confidence=opt.min_confidence,
        nms_max_overlap=opt.nms_max_overlap,
        min_detection_height=opt.min_detection_height,
        max_cosine_distance=opt.max_cosine_distance,
        nn_budget=opt.nn_budget,
        display=True
    )
    
    #end = time.time() - start
    #print(end)
