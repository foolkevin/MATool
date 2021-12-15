import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


datasets = ['LaSOT', 'OTB2015', 'GOT-10k']

class MotionAnalysis:
    def __init__(self, data_path, result_path, predict=False, predict_path=None, occ=False, ov=False):
        self.data_path = data_path
        self.result_path = result_path
        self.predict_path = predict_path
        self.predict = predict
        self.pathDict = self.get_files_path()
        self.occ = occ # Need to show occlusion case
        self.ov = ov # Nedd to show the case, out of view.
    
    def get_files_path(self):
        raise NotImplementedError

    def is_occ(self, occ_index):
        return occ_index == 1
    
    def is_ov(self, ov_index):
        return ov_index == 1
    
    def save_path(self, sequence):
        raise NotImplementedError
        
    def analyze_sequence(self, sequence):
        def set_axes(ax, title, index):
            ax.set_title(title, fontsize='xx-large', fontweight='bold')
            ax.plot(time_stamp, gt_bboxes[:, index], 'k.-', markersize=0.1)
            if self.predict:
                ax.plot(time_stamp, pred_bboxes[:, index], 'c*--', markersize=0.1)
            if self.occ:
                ax.fill_between(time_stamp, lowerborder[index], upborder[index], where=self.is_occ(occ_index), 
                                color='green', alpha=0.5)
            if self.ov:
                ax.fill_between(time_stamp, lowerborder[index], upborder[index], where=self.is_ov(ov_index), color='blue', 
                                alpha=0.5)
            if self.occ and self.ov:
                ax.text(0.95, 0.95, 'blue: out of view\ngreen: full of occlusion', horizontalalignment='center', 
                        verticalalignment='center', transform=ax.transAxes, fontweight='bold')
            elif self.occ:
                ax.text(0.95, 0.95, 'green: full of occlusion', horizontalalignment='center', 
                        verticalalignment='center', transform=ax.transAxes, fontweight='bold')
            elif self.ov:
                ax.text(0.95, 0.95, 'blue: out of view', horizontalalignment='center', 
                        verticalalignment='center', transform=ax.transAxes, fontweight='bold')
        
        def loadtxt(path):
            try:
                data = np.loadtxt(path)
            except:
                data = np.loadtxt(path, delimiter=',')
            return data
            
        seqinfo = self.pathDict[sequence]
        gt_bboxes = loadtxt(seqinfo['gt'])
        gt_bboxes[:, :2] = gt_bboxes[:, :2] + gt_bboxes[:, 2:] / 2
        if self.predict:
            pred_bboxes = loadtxt(seqinfo['pred'])
            pred_bboxes[:, :2] = pred_bboxes[:, :2] + pred_bboxes[:, 2:] / 2
        if self.occ:
            occ_index = loadtxt(seqinfo['occ'])
        if self.ov:
            ov_index = loadtxt(seqinfo['ov'])
        time_stamp = np.arange(0, gt_bboxes.shape[0])
        upborder1, lowerborder1 = gt_bboxes.max(axis=0), gt_bboxes.min(axis=0)
        if self.predict:
            upborder2, lowerborder2 = pred_bboxes.max(axis=0), pred_bboxes.min(axis=0)
            upborder, lowerborder = np.maximum(upborder1, upborder2), np.minimum(lowerborder1, lowerborder2)
        else:
            upborder, lowerborder = upborder1, lowerborder1
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(24, 24))
        set_axes(ax1, 'CenterX position', 0)
        set_axes(ax2, 'CenterY position', 1)
        set_axes(ax3, 'Width', 2)
        set_axes(ax4, 'Height', 3)
        save_path = self.save_path(sequence)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, sequence + '.jpg'))
        plt.close()

    def analyze(self):
        for sequence in tqdm(self.pathDict.keys(), '[Progress]', ncols=100):
            self.analyze_sequence(sequence)


class LaSOTAnalysis(MotionAnalysis):
    def __init__(self, data_path, result_path, predict=False, predict_path=None):
        super().__init__(data_path, result_path, predict, predict_path, occ=True, ov=True)
    
    def get_files_path(self):
        pathDict = dict()
        sequences = os.listdir(self.data_path) 
        for sequence in sequences:
            splits = os.listdir(os.path.join(self.data_path, sequence))
            for split in splits:
                pathDict[split] = {'gt': os.path.join(self.data_path, sequence, split, 'groundtruth.txt'),
                                   'occ': os.path.join(self.data_path, sequence, split, 'full_occlusion.txt'),
                                   'ov': os.path.join(self.data_path, sequence, split, 'out_of_view.txt')} 
                if self.predict:
                    pathDict[split]['pred'] = os.path.join(self.predict_path, sequence + '.txt')
        return pathDict
    
    def save_path(self, sequence):
        return os.path.join(self.result_path, 'LaSOT', sequence.split('-')[0])



class GOT10kAnalysis(MotionAnalysis):
    def __init__(self, data_path, result_path, predict=False, predict_path=None):
        super().__init__(data_path, result_path, predict, predict_path, occ=True, ov=True)
    
    def get_files_path(self):
        pathDict = dict()
        sequences = os.listdir(self.data_path)
        for sequence in sequences:
            pathDict[sequence] = {'gt': os.path.join(self.data_path, sequence, 'groundtruth.txt'),
                                  'occ': os.path.join(self.data_path, sequence, 'cover.label'),
                                  'ov': os.path.join(self.data_path, sequence, 'absence.label')}
            # Need to verify.
            if self.predict:
                pathDict[sequence]['pred'] = os.path.join(self.predict_path, sequence + '.txt')
        return pathDict 
    
    def is_occ(self, occ_index):
        return occ_index <= 4

    def save_path(self, sequence):
        return os.path.join(self.result_path, 'GOT-10k')


class OTBAnalysis(MotionAnalysis):
    def __init__(self, data_path, result_path, predict=False, predict_path=None):
        super().__init__(data_path, result_path, predict, predict_path, occ=False, ov=False)
    
    def get_files_path(self):
        pathDict = dict()
        sequences = os.listdir(self.data_path)
        for sequence in sequences:
            if sequence in ['Human4', 'Skating2', 'Jogging']:
                pathDict[sequence + '.1'] = {'gt': os.path.join(self.data_path, sequence, 'groundtruth_rect.1.txt')}
                pathDict[sequence + '.2'] = {'gt': os.path.join(self.data_path, sequence, 'groundtruth_rect.2.txt')}
                if self.predict:
                    pathDict[sequence + '.1']['pred'] = os.path.join(self.predict_path, sequence + '.1.txt')
                    pathDict[sequence + '.2']['pred'] = os.path.join(self.predict_path, sequence + '.2.txt')
            else:
                pathDict[sequence] = {'gt': os.path.join(self.data_path, sequence, 'groundtruth_rect.txt')}
                if self.predict:
                    pathDict[sequence]['pred'] = os.path.join(self.predict_path, sequence + '.txt')
        return pathDict

    def save_path(self, sequence):
        return os.path.join(self.result_path, 'OTB2015')


def parse():
    parser = argparse.ArgumentParser(description='A motion analysis tool for Object Tracking.')
    parser.add_argument('--dataset', type=str, choices=['LaSOT', 'OTB2015', 'GOT-10k'], help='The dataset which you want to analyze.')
    parser.add_argument('--data_path', type=str, help='The root path of the selected dataset.')
    parser.add_argument('--result_path', type=str, help='Where do you want to restore the analysis result.')
    parser.add_argument('--compared', action='store_true', help='Whether to compare gt with the prediction.')
    parser.add_argument('--predict_path', default=None, help='The root path where you fetch the prediction.')
    parser.add_argument('--video', type=str, default=None, help='Select a sequence to analyze.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    analysisDict = {'LaSOT': LaSOTAnalysis, 'GOT-10k': GOT10kAnalysis, 'OTB2015': OTBAnalysis}
    analyzer = analysisDict[args.dataset](args.data_path, args.result_path, args.compared, args.predict_path)
    if args.video:
        analyzer.analyze_sequence(args.video)
    else:
        analyzer.analyze()