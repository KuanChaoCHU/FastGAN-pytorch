# type: ignore
import os
import sys
import time
import datetime
import json
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

from . import pdf


def mkdirs(paths):
    """Create empty directories if they don't exist
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)
    return None


def mkdir(path):
    """Create a single
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_line_count(fpath):
    with open(fpath) as f:
        for i, line in enumerate(f):
            pass
    return i + 1


def timestamp():
    '''Returns the string-format timestamp for local time
    '''
    return time.strftime("%Y%m%d%H%M", time.localtime())          
        

def get_timestamp(with_brackets=True):
    time_result = time.strftime("%Y/%m/%d %H:%M", time.localtime())
    if with_brackets:
        return '[' + time_result + ']'
    else:
        return time_result


def save_to_pickle(path, target, overWrite=True):
    ''' Save the target object to path
    '''
    if not overWrite:
        if os.path.isfile(path):
            print("Pickle not saved. File already exists!")
            return 
    
    with open(path, 'wb') as f:
        pickle.dump(target, f)
    
    
def read_from_pickle(path):
    ''' Read from the .pickle file 
    '''
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_to_json():
    pass


def load_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data


def result_folder_cleaning(save_dir, folder_name, file_count_threshold=1,
                           wildcard=None):
    """ Automatically cleaning those result folders which has no. files lower 
        than given file_count_threshold. (empty experiment)
    Parameters:
        folder_name  (str)  Generally the date of experiments. e.g., 200527
        wildcard  (list)  If any of the string in this list exists in target folder,
                          the deletion of this folder will be passed.
                          (not implemented yet) 
    """
    assert file_count_threshold < 2
    target_folder = str(Path(save_dir) / folder_name)
    for subfolder in os.listdir(target_folder):
        sub_target_folder = str(Path(target_folder) / subfolder)
        sub_target_folder_files = os.listdir(sub_target_folder)
        
        #for w in wildcard:
        #    if w in sub_target_folder_files:
        #        break
        #else:
        if len(os.listdir(sub_target_folder)) <= file_count_threshold:
            print("Delete folder {}/{}: {} files in it.".format(folder_name, subfolder, 
                                                                len(sub_target_folder_files)))
            for file in sub_target_folder_files:
                del_fpath = str(Path(sub_target_folder) / file)
                os.remove(del_fpath)
            os.rmdir(sub_target_folder)
    print("Folder cleaning done.")


def merge_experiment_data(folder_path, new_name='report', to_pdf=False, 
                          **kwargs):
    """ Currently, output text is not aligned  
    """
    image_suffix = {'.jpg', '.png', '.bmp'}
    text_suffix = {'.txt'}
    images = []   
    all_lines = []
    
    for key in sorted(kwargs):
        if Path(kwargs[key]).suffix in image_suffix:
            images.append(str(Path(folder_path) / kwargs[key]))
        elif Path(kwargs[key]).suffix in text_suffix:        
            all_lines.append('{}\n'.format(kwargs[key]))
            with open(str(Path(folder_path) / kwargs[key])) as f:
                lines = f.readlines()
                all_lines.extend(lines)
            all_lines.append('\n')    
    
    new_name = '{}_{}_{}.txt'.format(new_name, Path(folder_path).parent.stem, 
                                     Path(folder_path).stem)
    with open(str(Path(folder_path) / new_name), 'w') as f:
        for line in all_lines:
            f.write(line)
            
    if to_pdf:
        pdf.pdf_gen(str(Path(folder_path) / 'report.pdf'), all_lines, images)


def generate_empty_config(option):
    """ Generate an empty .txt form of configurations 
    Hint: option=BaseOptions()
    """ 
    opt = option.parse()
    option.output_empty_file(opt)    
    

def rename_oldfile_if_exists(fpath):
    """ check file of same name exists or not.
        If exists, rename the old one with suffix "create time"
    """
    if os.path.exists(fpath):
        p = Path(fpath)
        name = p.stem
        format = p.suffix

        c_time = datetime.datetime.fromtimestamp(p.stat().st_ctime)
        c_time_timestamp = c_time.strftime("%y%m%d%H%M")
        new_fname = name + "_" + c_time_timestamp + format
        new_fpath = str(p.parent / new_fname)
        os.rename(fpath, new_fpath)
    # log this?


def write_log_to_file(log_object, fpath):
    """ Write the list of string logs to (.txt) file \n
    """
    rename_oldfile_if_exists(fpath)
    with open(fpath, 'w') as f:
        for line in log_object:
            f.write(str(line) + '\n') 


def get_layer_attr_suffix(layer):
    return f"_L{layer}" if layer in [1, 2] else ""


def get_layer_feature_d(opt, layer):
    """ Layer = [0, 1, 2]
    set "mixed" option to True if using concat feature (e.g., featL0L1) 
    """
    if not opt.is_feature_mixed or layer == 0:
        return opt.feature_d // (2**layer)
    else:  # NOW HARD-CODED
        dim_dct = {1: [1536, 512], 2: [768, 512]}
        if layer == 1: return dim_dct[1][opt.stage_L1]
        elif layer == 2: return dim_dct[2][opt.stage_L2]


def get_layer_grid_size(input_size, layer):
    """ Returns the featmap grid size 
    """
    return input_size // (32 // (2**layer))


def get_mix_suffix(opt, layer):
    """ e.g., "", "_mixed_s1", "_mixed_s10"
    """
    mixed_dct = {True: '_mixed', False: ''}
    if not opt.is_feature_mixed or layer == 0:
        stage = ''
    elif layer == 1:
        stage = '_s' + str(opt.stage_L1)
    elif layer == 2:
        stage = '_s' + str(opt.stage_L1) + str(opt.stage_L2)
    return mixed_dct[opt.is_feature_mixed] + stage


def printlog(message, log_object=None):
    """ Print message to console and save to log if provided.
    """
    print(message)
    if log_object is not None:
        try:
            log_object.add_event(message)
        except AttributeError:
            try:  # better style?
                log_object.append(get_timestamp() + message)
            except:
                print("Incompatible log_object. Expect a ExpManager or list.")
        

def corner2center_t(corner_boxes_t):
    """ Convert the corner type box tensor to center type box tensor
    [xmin, ymin, xmax, ymax] -> [xcenter, ycenter, w, h]
    """        
    device = corner_boxes_t.device
    center_boxes_t = torch.full(corner_boxes_t.size(), fill_value=-1.0,
                                dtype=torch.float32, device=device)
    center_boxes_t[:, 2:4] = corner_boxes_t[:, 2:4] - corner_boxes_t[:, 0:2]
    center_boxes_t[:, 0:2] = corner_boxes_t[:, 0:2] + center_boxes_t[:, 2:4] * 0.5    
    return center_boxes_t

    
def center2corner(center_boxes):
    # was in m_scenegraph.data.coco
    # center_xywh: ndarray (n, 4)
    #              [0,1] x_center, y_center, w, h    (ytrue of rescaled image)
    # corner_xywh: [0,1] x_min, y_min, x_max, y_max  (sgbbox original image?)
    corner_boxes = np.full(shape=center_boxes.shape, fill_value=-1.0, dtype='float32')
    corner_boxes[:, 0:2] = center_boxes[:, 0:2] - center_boxes[:, 2:4] * 0.5
    corner_boxes[:, 2:4] = center_boxes[:, 0:2] + center_boxes[:, 2:4] * 0.5
    return corner_boxes


#TODO: Merge to the ExpManager class    
class ExpInfo:
    """ Only used in base_options, to generate the exp_name
    
    """
    def __init__(self, opt):
        self.save_folder = opt.save_dir
        self.exp_date = timestamp()[2:8]
        if opt.create_exp:
            mkdir(str(Path(self.save_folder) / self.exp_date))
            
            exp_max = 0
            for x in os.listdir(str(Path(self.save_folder) / self.exp_date)):            
                if os.path.isdir(str(Path(self.save_folder) / self.exp_date / x)) and \
                  x.startswith(opt.model):
                    exp_max = max(exp_max, int(x.split('_')[1]))
            self.exp_name = opt.model + '_' + str(exp_max + 1)       
            self.exp_dir = str(Path(self.save_folder) / self.exp_date / self.exp_name)
            mkdir(self.exp_dir)        
        else:
            self.exp_dir = None
            

class MiniTimer():
    def __init__(self, cuda_sync=True, digits=2):
        self.is_sync = cuda_sync                # call synchronize()
        self.digits = digits                    # round
        self.t_checkpoint = defaultdict(float)  # {codename: ts} 
        self.t_elapse = defaultdict(float)      # {codename: t_total}
        self.topics = []                        # [(name, codename)]

    def start(self, name):
        matched = [t for t in self.topics if t[0] == name]
        if matched:  codename = matched[0][1]            
        else:
            codename = 't{}'.format(len(self.topics)+1)
            self.topics.append((name, codename))        
        if self.is_sync and torch.cuda.is_available():
            torch.cuda.synchronize()            
        self.t_checkpoint[codename] = time.time()
        return codename

    def end(self, codename):
        if self.is_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t_elapse[codename] += (time.time() - self.t_checkpoint[codename])    
        self.t_checkpoint[codename] = 0.0
    
    def print(self):
        print("Time comsumption: ")
        for name, codename in self.topics:
            print("{}({}): {}".format(name, codename, 
                                      round(self.t_elapse[codename], self.digits)))

