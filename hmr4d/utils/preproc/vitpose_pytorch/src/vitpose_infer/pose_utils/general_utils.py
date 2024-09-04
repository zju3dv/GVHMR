#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:49:22 2022

@author: gpastal
"""
import numpy as np
import numpy.ma as ma
# import pika
import json
from collections import OrderedDict
from collections.abc import Iterable
from itertools import chain
import argparse

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    # exp file
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.2, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=240, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

def jitter(tracking,temp,id1):
    pass
def jitter2(tracking,temp,id1)  :
    pass
    
def create_json_rabbitmq( FRAME_ID,pose):
    pass

def producer_rabbitmq():
    pass
def fix_head(xyz):
    pass

def flatten_lst(x):
    if isinstance(x, Iterable):
        return [a for i in x for a in flatten_lst(i)]
    else:
        return [x]

def polys_from_pose(pts):
    seg=[]
    for ind, i in enumerate(pts):
        list_=[]
        list_sc=[]
        # list1 = [i[0][1],i[0][0]]

        # list2 = [i[0][1],i[0][0]]
        # print(i)
        for j in i:
          
            temp_ = [j[1],j[0]]
            
            if j[2]>0.4:

                temp2_ = [1]
            else:
                temp2_ =[0]
            list_.append(temp_)
            list_sc.append(temp2_)
            # print(list_sc)
            # list2 = [i[6][1],i[6][0]]
            # list3 = [i[11][1],i[11][0]]
            # list4 = [i[12][1],i[12][0]]
        
        # list_ = flatten_lst(list_)
        # print(list_)
        list_=fix_list_order(list_,list_sc)
        # print(list_)
        # list_=list(list_)
        # list_ = list_.to_list()
        # print(list_)
        # temp__=list(chain(*list_))
        seg.append(list_)   # temp_ = list(chain(list1,list2,list3,list4,list1))
    return seg
def fix_list_order(list_,list2):
    # for index,values in enumerate(list_):
    myorder = [0, 2, 4, 6, 8,10,12,14,16,15,13,11,9,7,5,3,1]
    cor_list = [list_[i] for i in myorder]
    cor_list2 = [list2[i] for i in myorder]
    # print(cor_list)
    # result = list(set(map(tuple,cor_list)) & set(map(tuple,cor_list2)))
    # arr = np.array([x for x in cor_list])
    # print(cor_list)
    data = np.asarray(cor_list)
    # print(data)
    mask = np.column_stack((cor_list2, cor_list2))
    # masked = ma.masked_array(data, mask=np.column_stack((cor_list2, cor_list2)))#[cor_list2,cor_list2])
    # result = list(set(masked[~masked.mask]))
    # print(result)
    # print(data)
    # print(mask)
    result2 = []
    for inde,i in enumerate(data):
        # print(mask[inde])
        if mask[inde].all()==1:
            result2.append(i[0])
            result2.append(i[1])
    # result = [int(result[i] for i in result)]
    return result2

