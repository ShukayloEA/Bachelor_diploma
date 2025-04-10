import json
import argparse
from os.path import join
'''
data = {
    'Anti-UAV-RGBT': {
        'train':[
            '20190925_101846_1_1',
            '20190925_101846_1_2'
        ],
        'test':[
            '20190925_111757_1_1\\infrared.mp4',
            '20190925_111757_1_1',
            '20190925_111757_1_2'
        ]
    }
}
'''
class opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        '''
        self.parser.add_argument(
            'dataset',
            type=str,
            help='Anti-UAV-RGBT',
        )
        '''
        self.parser.add_argument(
            'file',
            type=str,
            help='Enter mp4 file or directory with frames',
        )
        self.parser.add_argument(
            'mode',
            type=str,
            help='infrared or visible',
        )
        self.parser.add_argument(
            '--NSA',
            action='store_true',
            help='NSA Kalman filter'
        )
        self.parser.add_argument(
            '--EMA',
            action='store_true',
            help='EMA feature updating mechanism'
        )
        self.parser.add_argument(
            '--MC',
            action='store_true',
            help='Matching with both appearance and motion cost'
        )
        self.parser.add_argument(
            '--root_dataset',
            default='./'
        )
        self.parser.add_argument(
            '--dir_save',
            default='./'
        )
        self.parser.add_argument(
            '--EMA_alpha',
            default=0.9
        )
        self.parser.add_argument(
            '--MC_lambda',
            default=0.98
        )

    def parse(self, args=''):
        if args == '':
          opt = self.parser.parse_args()
        else:
          opt = self.parser.parse_args(args)
        opt.min_confidence = 0.5
        opt.nms_max_overlap = 0.7
        opt.min_detection_height = 0
        opt.max_cosine_distance = 0.3
        if opt.MC:
            opt.max_cosine_distance += 0.05
        if opt.EMA:
            opt.nn_budget = 1
        else:
            opt.nn_budget = 100
        opt.sequence = opt.file
        opt.is_infrared = True if opt.mode == 'infrared' else False
        '''
        opt.dir_dataset = join(
            opt.root_dataset,
            opt.dataset,
            opt.mode
            #'train' if opt.mode == 'val' else 'test'
        )
        '''
        return opt

opt = opts().parse()