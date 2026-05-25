import argparse
import os
import json
from pathlib import Path
import traceback
from typing import List, Optional

import pandas as pd
import torch
from filelock import FileLock
from hamer.configs import dataset_eval_config
from hamer.datasets import create_dataset
from hamer.utils import Evaluator, recursive_to
from tqdm import tqdm

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--results_folder', type=str, default='results', help='Path to results folder.')
    parser.add_argument('--dataset', type=str, default='FREIHAND-VAL,HO3D-VAL,NEWDAYS-TEST-ALL,NEWDAYS-TEST-VIS,NEWDAYS-TEST-OCC,EPICK-TEST-ALL,EPICK-TEST-VIS,EPICK-TEST-OCC,EGO4D-TEST-ALL,EGO4D-TEST-VIS,EGO4D-TEST-OCC', help='Dataset to evaluate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of test samples to draw')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers used for data loading')
    parser.add_argument('--log_freq', type=int, default=10, help='How often to log results')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False, help='Shuffle the dataset during evaluation')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HMR2.0 model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load config and run eval, one dataset at a time
    print('Evaluating on datasets: {}'.format(args.dataset), flush=True)
    for dataset in args.dataset.split(','):
        dataset_cfg = dataset_eval_config()[dataset]
        args.dataset = dataset
        run_eval(model, model_cfg, dataset_cfg, device, args)

def run_eval(model, model_cfg, dataset_cfg, device, args):

    # List of metrics to log
    if args.dataset in ['FREIHAND-VAL', 'HO3D-VAL']:
        metrics = None
        preds = ['vertices', 'keypoints_3d']
        pck_thresholds = None
        rescale_factor = -1
    elif args.dataset in ['NEWDAYS-TEST-ALL', 'NEWDAYS-TEST-VIS', 'NEWDAYS-TEST-OCC',
                          'EPICK-TEST-ALL', 'EPICK-TEST-VIS', 'EPICK-TEST-OCC',
                          'EGO4D-TEST-ALL', 'EGO4D-TEST-VIS', 'EGO4D-TEST-OCC']:
        metrics = ['mode_kpl2']
        preds = None
        pck_thresholds = [0.05, 0.1, 0.15]
        rescale_factor = 2

    # Create dataset and data loader
    dataset = create_dataset(model_cfg, dataset_cfg, train=False, rescale_factor=rescale_factor)
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    # Setup evaluator object
    evaluator = Evaluator(
        dataset_length=dataset.__len__(),
        dataset=args.dataset,
        keypoint_list=dataset_cfg.KEYPOINT_LIST, 
        pelvis_ind=model_cfg.EXTRA.PELVIS_IND, 
        metrics=metrics,
        preds=preds,
        pck_thresholds=pck_thresholds,
    )

    # Go over the images in the dataset.
    try:
        for i, batch in enumerate(tqdm(dataloader)):
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
            evaluator(out, batch)
            if i % args.log_freq == args.log_freq - 1:
                evaluator.log()
        evaluator.log()
        error = None
    except (Exception, KeyboardInterrupt) as e:
        traceback.print_exc()
        error = repr(e)
        i = 0

    # Append results to file
    if metrics is not None:
        metrics_dict = evaluator.get_metrics_dict()
        results_csv = os.path.join(args.results_folder, 'eval_regression.csv')
        save_eval_result(results_csv, metrics_dict, args.checkpoint, args.dataset, error=error, iters_done=i, exp_name=args.exp_name)
    if preds is not None:
        results_json = os.path.join(args.results_folder, '%s.json' % args.dataset.lower())
        preds_dict = evaluator.get_preds_dict()
        save_preds_result(results_json, preds_dict)

def save_eval_result(
    csv_path: str,
    metric_dict: float,
    checkpoint_path: str,
    dataset_name: str,
    # start_time: pd.Timestamp,
    error: Optional[str] = None,
    iters_done=None,
    exp_name=None,
) -> None:
    """Save evaluation results for a single scene file to a common CSV file."""

    timestamp = pd.Timestamp.now()
    exists: bool = os.path.exists(csv_path)
    exp_name = exp_name or Path(checkpoint_path).parent.parent.name

    # save each metric as different row to the csv path
    metric_names = list(metric_dict.keys())
    metric_values = list(metric_dict.values())
    N = len(metric_names)
    df = pd.DataFrame(
        dict(
            timestamp=[timestamp] * N,
            checkpoint_path=[checkpoint_path] * N,
            exp_name=[exp_name] * N,
            dataset=[dataset_name] * N,
            metric_name=metric_names,
            metric_value=metric_values,
            error=[error] * N,
            iters_done=[iters_done] * N,
        ),
        index=list(range(N)),
    )

    # Lock the file to prevent multiple processes from writing to it at the same time.
    lock = FileLock(f"{csv_path}.lock", timeout=10)
    with lock:
        df.to_csv(csv_path, mode="a", header=not exists, index=False)

def save_preds_result(
    pred_out_path: str,
    preds_dict: float,
) -> None:
    """ Save predictions into a json file. """
    xyz_pred_list = preds_dict['keypoints_3d']
    verts_pred_list = preds_dict['vertices']
    # make sure its only lists
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]

    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump(
            [
                xyz_pred_list,
                verts_pred_list
            ], fo)
    print('Dumped %d joints and %d verts predictions to %s' % (len(xyz_pred_list), len(verts_pred_list), pred_out_path))

if __name__ == '__main__':
    main()
