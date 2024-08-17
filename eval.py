import argparse
import json
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
import torch

from omni3d_evaluation import Omni3DEvaluationHelper
from datasets import _load_omni3d_json


def evaluate(name: str, gt_ann_path: str, pred_ann_path: str, log_dir: str) -> None:
    filter_settings = {
        'category_names': None,
        'ignore_names': [],
        'truncation_thres': 1 / 3,
        'visibility_thres': 1 / 3,
        'min_height_thres': 0.02,
        'modal_2D_boxes': False,
        'trunc_2D_boxes': True,
        'max_depth': 1e5,
        'max_height_thres': 1.5
    }
    iou_thres = (0.5, 0.5)

    setup_logger(output=log_dir, name="default")

    # Load Ground Truth
    gt_data = json.load(open(gt_ann_path, 'r'))
    sorted_cats = sorted(gt_data['categories'], key=lambda cat: cat['id'])
    cat_ids = [cat['id'] for cat in sorted_cats]
    cat_names = [cat['name'] for cat in sorted_cats]
    MetadataCatalog.get('omni3d_model').thing_classes = cat_names
    MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id = {id: id for id in cat_ids}
    filter_settings['category_names'] = cat_names
    _load_omni3d_json(gt_ann_path, '', name, filter_settings, filter_empty=False)
    MetadataCatalog.get(name).set(json_file=gt_ann_path, image_root='', evaluator_type='coco')

    # Load Preditions
    eval_helper = Omni3DEvaluationHelper([name], filter_settings, output_folder='', iter_label=0, only_2d=False,
                                         iou_3d_thresholds_range=tuple(iou_thres))
    pred_results = torch.load(pred_ann_path)
    eval_helper.add_predictions(name, pred_results)

    # Evaluate
    eval_helper.evaluate(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--gt_ann', type=str, required=True, dest="gt_ann_path")
    parser.add_argument('--pred_ann', type=str, required=True, dest="pred_ann_path")
    parser.add_argument('--log_dir', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(**vars(args))
