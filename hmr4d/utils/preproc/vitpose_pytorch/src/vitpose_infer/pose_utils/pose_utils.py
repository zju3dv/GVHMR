#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:45:33 2022

@author: gpastal
"""
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms as TR
import numpy as np
import cv2
import logging
# from  simpleHRNet.models_.hrnet import HRNet
# from torch2trt import torch2trt,TRTModule
logger = logging.getLogger("Tracker !")
from .timerr import Timer
from pathlib import Path
# import gdown
timer_det = Timer()
timer_track = Timer()
timer_pose = Timer()

def pose_points_yolo5(detector,image,pose,tracker,tensorrt):
            timer_det.tic()
            # starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
           
            transform = TR.Compose([
                TR.ToPILImage(),
                # Padd(),
                TR.Resize((256, 192)),  # (height, width)
                TR.ToTensor(),
                TR.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            # image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            detections = detector(image)
            timer_det.toc()
            logger.info('DET FPS -- %s',1./timer_det.average_time)
            # print(detections.shape)
            dets = detections.xyxy[0]
            dets = dets[dets[:,5] == 0.]
            # dets = dets[dets[:,4] > 0.3]
            # logger.warning(len(dets))
            
            # if len(dets)>0:
                # image_gpu = torch.tensor(image).cuda()/255.
                # print(image_gpu.size())
            timer_track.tic()
            online_targets=tracker.update(dets,[image.shape[0],image.shape[1]],image.shape)

            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                # vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_threshs
                if tlwh[2] * tlwh[3] > 10 :#and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # tracker.update()
            timer_track.toc()
            logger.info('TRACKING FPS --%s',1./timer_track.average_time)
            device='cuda'
            nof_people = len(online_ids) if online_ids is not None else 0
            # nof_people=1
            # print(dets)
            # print(nof_people)
            boxes = torch.empty((nof_people, 4), dtype=torch.int32,device= 'cuda')
            # boxes = []
            images = torch.empty((nof_people, 3, 256, 192))  # (height, width)
            heatmaps = np.zeros((nof_people, 17, 64, 48),
                                dtype=np.float32)
            # starter.record()
            # print(online_tlwhs)
            if len(online_tlwhs):
                for i, (x1, y1, x2, y2) in enumerate(online_tlwhs):
                # for i, (x1, y1, x2, y2) in enumerate(np.array([[55,399,424-55,479-399]])):
                # if i<1:
                    x1 = x1.astype(np.int32)
                    x2 = x1+x2.astype(np.int32)
                    y1 = y1.astype(np.int32)
                    y2 = y1+ y2.astype(np.int32)
                    if x2>image.shape[1]:x2=image.shape[1]-1
                    if y2>image.shape[0]:y2=image.shape[0]-1
                    if y1<0: y1=0
                    if x1<0 : x1=0
                    # print([x1,x2,y1,y2])
                    # image = cv2.rectangle(image, (x1,y1), (x2,y2), (0,0,0), 1)
            # cv2.imwrite('saved.png',image)
            #         # Adapt detections to match HRNet input aspect ratio (as suggested by xtyDoge in issue #14)
                    correction_factor = 256 / 192 * (x2 - x1) / (y2 - y1)
                    if correction_factor > 1:
                        # increase y side
                        center = y1 + (y2 - y1) // 2
                        length = int(round((y2 - y1) * correction_factor))
                        y1_new = int( center - length // 2)
                        y2_new = int( center + length // 2)
                        image_crop = image[y1:y2, x1:x2, ::-1]
                        # print(y1,y2,x1,x2)
                        pad = (int(abs(y1_new-y1))), int(abs(y2_new-y2))
                        image_crop = np.pad(image_crop,((pad), (0, 0), (0, 0)))
                        images[i] = transform(image_crop)
                        boxes[i]= torch.tensor([x1, y1_new, x2, y2_new])
                    
                    elif correction_factor < 1:
                        # increase x side
                        center = x1 + (x2 - x1) // 2
                        length = int(round((x2 - x1) * 1 / correction_factor))
                        x1_new = int( center - length // 2)
                        x2_new = int( center + length // 2)
                        # images[i] = transform(image[y1:y2, x1:x2, ::-1])
                        image_crop = image[y1:y2, x1:x2, ::-1]
                        pad = (abs(x1_new-x1)), int(abs(x2_new-x2))
                        image_crop = np.pad(image_crop,((0, 0), (pad), (0, 0)))
                        images[i] = transform(image_crop)
                        boxes[i]= torch.tensor([x1_new, y1, x2_new, y2])
                        

                if images.shape[0] > 0:
                        images = images.to(device)

                        if tensorrt:
                            out = torch.zeros((images.shape[0],17,64,48),device=device)
                            with torch.no_grad():
                                timer_pose.tic()

                                for i in range(images.shape[0]):
                                    # timer_pose.tic()
                                    # print(images[i].size())
                                    
                                    out[i] = pose(images[i].unsqueeze(0))
                                timer_pose.toc()
                                logger.info('POSE FPS -- %s',1./timer_pose.average_time)
                        else:
                            with torch.no_grad():
                                
                                timer_pose.tic()

                        
                                    
                                out = pose(images)
                                timer_pose.toc()
                                logger.info('POSE FPS -- %s',1./timer_pose.average_time)
                            

                        pts = torch.empty((out.shape[0], out.shape[1], 3), dtype=torch.float32,device=device)
                        pts2 = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)

                        (b,indices)=torch.max(out,dim=2)
                        (b,indices)=torch.max(b,dim=2)
                        
                        (c,indicesc)=torch.max(out,dim=3)
                        (c,indicesc)=torch.max(c,dim=2)
                        dim1= torch.tensor(1. / 64,device=device)
                        dim2= torch.tensor(1. / 48,device=device)

                        for i in range(0,out.shape[0]):

                                pts[i, :, 0] = indicesc[i,:] * dim1 * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                                pts[i, :, 1] = indices[i,:] *dim2* (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                                pts[i, :, 2] = c[i,:]

                        pts=pts.cpu().numpy()
                    # print(pts)
            else:
                pts = np.empty((0, 0, 3), dtype=np.float32)
                online_tlwhs = []
                online_ids = []
                online_scores=[]
            res = list()

            res.append(pts)


            if len(res) > 1:
                return res,online_tlwhs,online_ids,online_scores#,pts2
            else:
                return res[0],online_tlwhs,online_ids,online_scores#,pts2

