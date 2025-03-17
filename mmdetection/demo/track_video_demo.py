# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from botsort.yolox.utils.visualize import plot_tracking

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
from botsort.tracker.mc_bot_sort import BoTSORT
from botsort.tracker.tracking_utils.timer import Timer
import numpy as np
import os.path as osp
import time

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # build test pipeline
    model.cfg.test_dataloader.dataset.pipeline[
        0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    video_reader = mmcv.VideoReader(args.video)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))
    
    t_args = argparse.Namespace(
        ablation=False, 
        appearance_thresh=0.25, 
        camid=0, 
        ckpt='/ext/hoya/bytetrack_x_mot17.pth.tar', 
        cmc_method='sparseOptFlow', 
        conf=None, demo='video', 
        device='gpu', 
        exp_file='/mmdetection/botsort/yolox/exps/example/mot/yolox_x_mix_det.py', 
        experiment_name='yolox_x_mix_det', 
        fast_reid_config='/mmdetection/botsort/fast_reid/configs/MOT17/sbs_S50.yml', 
        fast_reid_weights='/ext/hoya/mot17_sbs_S50.pth', 
        fp16=True, 
        fps=30, 
        fuse=True, 
        fuse_score=True, 
        match_thresh=0.8, 
        min_box_area=10, 
        mot20=False, 
        name=None, 
        new_track_thresh=0.7, #이게 낮으면 많이 나오긴하는 듯
        nms=None, 
        path='/ext/mot_sample_video/cctv_sample.mp4', 
        proximity_thresh=0.5, 
        save_result=True, 
        track_buffer=30, 
        track_high_thresh=0.3, 
        track_low_thresh=0.1, 
        trt=False, 
        tsize=None, 
        with_reid=True
    )
    #print(t_args)
    tracker = BoTSORT(t_args, frame_rate=video_reader.fps)
    vid_writer = cv2.VideoWriter(
        '/bot_sort_test.mp4', cv2.VideoWriter_fourcc(*"mp4v"), video_reader.fps, (int(video_reader.width), int(video_reader.height))
    )
    timer = Timer()
    frame_id = 0
    results = []
    for frame in track_iter_progress((video_reader, len(video_reader))):
        #print(frame, len(frame))
        #print(frame.shape)
        result = inference_detector(model, frame, test_pipeline=test_pipeline)
        pre_instances = result.pred_instances
        #print(result)
        outputs = np.empty((1,7))
        #print(pre_instances)
        for i, instancedata in enumerate(pre_instances):
            output = np.array([])
            left, bottom, right, top = map(round, instancedata.bboxes.tolist()[0])
            #left,top,right,bottom
            output = np.append(output, left)
            output = np.append(output, bottom)
            output = np.append(output, right)
            output = np.append(output, top)
            score = instancedata.scores.tolist()[0]
            label = instancedata.labels.tolist()[0]
            
            output = np.append(output, score)
            output = np.append(output, 1)
            output = np.append(output, label)
            outputs = np.vstack((outputs,output))
        #print(outputs, type(outputs), len(outputs))
        #print(outputs)
        online_targets = tracker.update(outputs[1:], frame)
    
        #print(online_targets)
        
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            if tlwh[2] * tlwh[3] > 10:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                results.append(
                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                )
    
        timer.toc()
        online_im = plot_tracking(
                frame, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
        )
        vid_writer.write(online_im)
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
        
        frame_id += 1
        #timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        
        
        
    res_file = osp.join("/demo_tracking.txt")
    with open(res_file, 'w') as f:
        f.writelines(results)       
        
    
        #scale = min(800 / float(ori_shape[0], ), 1440 / float(ori_shape[1]))
        
        # detections = []
        # if outputs[0] is not None:
        #     outputs = outputs[0].cpu().numpy()
        #     detections = outputs[:, :7]
        #     detections[:, :4] /= scale

        
    #     visualizer.add_datasample(
    #         name='video',
    #         image=frame,
    #         data_sample=result,
    #         draw_gt=False,
    #         show=False,
    #         pred_score_thr=args.score_thr)
    #     frame = visualizer.get_image()

    #     if args.show:
    #         cv2.namedWindow('video', 0)
    #         mmcv.imshow(frame, 'video', args.wait_time)
    #     if args.out:
    #         video_writer.write(frame)

    # if video_writer:
    #     video_writer.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
