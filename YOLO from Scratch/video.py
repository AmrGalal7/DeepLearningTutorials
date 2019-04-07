from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from utils import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random


def arg_parse():
    """
    parses arguments to the detect module

    """

    parser = argparse.ArgumentParser(description = 'YOLO v3 Detection Module')
    parser.add_argument("--det", dest = 'det',
                        help = "images directory to store detections in", default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "object confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = "configuration file path", default = "config/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = "weights file path", default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso',
                        help = "input resolution of the network. increase to increase accuracy. decrease to increase speed", default = "416", type = str)
    parser.add_argument("--video", dest = "videofile", help = "video file to run detection on", default = "video.avi", type = str)

    return parser.parse_args()


args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

classes = load_classes("data/coco.names")
# num_classes = 80 for COCO dataset
num_classes = len(classes)

print('loading the model...')
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print('model loaded successfully')

model.net_info["height"] = args.reso
input_dim = int(model.net_info["height"])
assert input_dim % 32 == 0
assert input_dim > 32

# if there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

# set the model in evaluation mode
model.eval()


def draw_bbox(bbox, image):
    # get the bbox centre (x, y)
    top_left_corner = tuple(bbox[1:3].int())
    # get the bbox width and height
    box_dim = tuple(bbox[3:5].int())

    cls = int(bbox[-1])
    label = '{}'.format(classes[cls])
    colour = random.choice(colours)
    cv2.rectangle(image, top_left_corner, box_dim, colour, 1)

    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    text_dim = top_left_corner[0] + t_size[0] + 3, top_left_corner[1] + t_size[1] + 4
    # draw a filled rectangle for the text
    cv2.rectangle(image, top_left_corner, text_dim, colour, -1)
    cv2.putText(image, label, (top_left_corner[0], top_left_corner[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)

    return image


videofile = args.videofile

# for webcame
if videofile == '0':
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(videofile)


assert cap.isOpened(), 'Cannot capture source'

frames = 0
start = time.time()

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        img = prep_image(frame, input_dim)
        # cv2.imshow("a", frame)
        img_dim = frame.shape[1], frame.shape[0]
        img_dim = torch.FloatTensor(img_dim).repeat(1, 2)

        if CUDA:
            img_dim = img_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = model(Variable(img), CUDA)

        output = write_results(output, confidence, num_classes, nms_thesh)

        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue

        img_dim = img_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416/img_dim,1)[0].view(-1,1)

        output[:,[1,3]] -= (input_dim - scaling_factor*img_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (input_dim - scaling_factor*img_dim[:,1].view(-1,1))/2

        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, img_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, img_dim[i,1])

        classes = load_classes('data/coco.names')
        colours = pkl.load(open("palette", "rb"))

        list(map(lambda x: draw_bbox(x, frame), output))

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1
        print(time.time() - start)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

    else:
        break
