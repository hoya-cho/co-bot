import os
import argparse
import json

from gtgen.tools.libs.aimmo2coco.box import AimmoGT2CocoBox


def get_args():
    parser = argparse.ArgumentParser('Convert to COCO Json Format')
    parser.add_argument('-a', '--aimmo_gt', type=str)
    parser.add_argument('-d', '--dst', type=str)
    parser.add_argument('-dr', '--dataset_root_dir', type=str)
    parser.add_argument('-ih', '--img_height', type=int, default=None)
    parser.add_argument('-iw', '--img_width', type=int, default=None)
    args = parser.parse_args()
    return args


def main(args):
    img_height = args.img_height
    img_width = args.img_width
    if (img_height and img_width):
        img_size = (img_height, img_width)
    else:
        img_size = None

    INITIAL_SETTINGS_PATH = '/2dod-torch/gtgen/tools/class_map.json'
    INIT_JSON = json.load(open(INITIAL_SETTINGS_PATH))
    print(INIT_JSON)
    CLASSES_MAP = INIT_JSON.get('class_map') 
    print(CLASSES_MAP)
    cls_map = {int(cls_num): cls_name for cls_num, cls_name in CLASSES_MAP.items()}
    
    
    print('\n\n====== Convert AIMMO GT bbox to COCO Format ======\n')
    AimmoGT2CocoBox(
        
        cls_map= cls_map,
        aimmo_gt_path=args.aimmo_gt,
        dataset_root_dir=args.dataset_root_dir,
        img_size=img_size,
    ).save(dst=args.dst)
    print(f'\n*** >>> Completed. Output path is [{os.path.abspath(args.dst)}]) ***\n\n')
    

if __name__ == '__main__':
    args = get_args()
    main(args=args)