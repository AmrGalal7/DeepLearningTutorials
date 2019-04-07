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
    parser.add_argument("--images", dest = 'images',
                        help = "image directory containing images to perform detection upon", default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det',
                        help = "images directory to store detections in", default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "object confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = "configuration file path", default = "config/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = "weights file path", default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso',
                        help = "input resolution of the network. increase to increase accuracy. decrease to increase speed", default = "416", type = str)
    return parser.parse_args()


args = arg_parse()
images = args.images
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

read_dir = time.time()
# detection phase
try:
    imgs_list = [ osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) ]
except NotADirectoryError:
    imgs_list = []
    imgs_list.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print('No file or directory with name {}'.format(images))
    exit()

# if the detection directory 'det' doesn't exist, create it
if not os.path.exists(args.det):
    os.makedirs(args.det)

load_batch = time.time()
loaded_imgs = [ cv2.imread(x) for x in imgs_list ]

# pytorch variables for the images
img_batches = list( map(prep_image, loaded_imgs, [ input_dim for x in range(len(imgs_list)) ]) )



####################### display the padded image ###############################
# image= img_batches[0].squeeze(0).data.numpy().transpose((1, 2, 0))[:, :, ::-1]
#
# cv2.imshow('im', image)
# # Pause here 5 seconds.
# k = cv2.waitKey(5000)
#
# if k == 27:         # If escape was pressed exit
#     cv2.destroyAllWindows()
################################################################################


# list of the dimensions of the original images
orig_imgs_dim_list = [ (x.shape[1], x.shape[0]) for x in loaded_imgs ]
# why repeat?
orig_imgs_dim_list = torch.FloatTensor(orig_imgs_dim_list).repeat(1, 2)


if CUDA:
    orig_imgs_dim_list = orig_imgs_dim_list.cuda()

# create the batches
leftover = 0
if(len(orig_imgs_dim_list) % batch_size):
    leftover = 1
if batch_size != 1:
    num_batches = len(imgs_list) // batch_size + leftover
    img_batches = [ torch.cat( img_batches[ i * batch_size : min((i + 1)*batch_size, len(img_batches)) ] ) \
                    for i in range(num_batches)]

# detection loop
write = 0
start_det_loop = time.time()


for i, batch in enumerate(img_batches):

    # load the image
    start = time.time()
    if CUDA:
        batch = batch.cuda()

    with torch.no_grad():
        # no need for 'Volatile' wrapper?
        prediction = model(Variable(batch), CUDA)
    prediction = write_results(prediction, confidence, num_classes, nms_confidence = nms_thesh)

    end = time.time()

    # if 'write_results' outputs int(0) signifying there's no detection, skip the rest of this loop
    # prediction.type() instead?
    if(type(prediction)) == int:

        for img_num, image in enumerate( imgs_list[i * batch_size : min((i+1)*batch_size, len(imgs_list))] ):
            img_id = i * batch_size + img_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")

        continue


    # transform the atribute from index in 'batch' to index in 'imgs_list'
    prediction[:, 0] += i * batch_size

    if not write:
        output = prediction
        write = 1
    else:
        output = torch.cat( (output, prediction) )

    for img_num, image in enumerate(imgs_list[ i*batch_size : min((i+1)*batch_size, len(imgs_list)) ]):
        img_id = i * batch_size + img_num
        # from all the bboxes, get the ones that belong to the image 'img_id',
        # then out of the remaining bboxes, get the predicted class of each
        objs = [ classes[int(x[-1])] for x in output if int(x[0]) == img_id ]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")


    # makes sure that CUDA kernel is synchronised with the CPU.
    if CUDA:
        torch.cuda.synchronize()

try:
    output
except NameError:
    print ("No detections were made")
    exit()

# transform the co-ordinates of the bboxes to be measured w.r.t. the boundaries
# of the area on the padded image that contains the original image:
# among the original dimensions of all the input images, select those that the model has predictions for.
orig_imgs_dim_list = torch.index_select( orig_imgs_dim_list, 0, output[:, 0].long() )



# get the scaling factor for each image --> (model's input shape / image shape)
#                                       --> this is the same value used in padding.
# 416: the model's input shape?
# [0] to select the values, not the args from the return os 'torch.min'
scaling_factor = torch.min( 416/orig_imgs_dim_list, 1 )[0].view(-1, 1)

# map the prediction from the orginal image size to the padded area (shift the padded distance, then scale)
# -input_dim?
output[:, [1, 3]] -= ( input_dim - scaling_factor * orig_imgs_dim_list[:, 0].view(-1, 1)) / 2
output[:, [2, 4]] -= ( input_dim - scaling_factor * orig_imgs_dim_list[:, 1].view(-1, 1)) / 2
output[:, 1:5] /= scaling_factor

# clip any bboxes having boundaries outside the image to the edges of our image
for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp( output[i, [1, 3]], 0.0, orig_imgs_dim_list[i, 0] )
    output[i, [2, 4]] = torch.clamp( output[i, [2, 4]], 0.0, orig_imgs_dim_list[i, 1] )


output_recast = time.time()

# drawing the bboxes:
class_loaded = time.time()
# load the pickled file 'palette' that contains many colours
with open('palette', 'rb') as f:
    colours = pkl.load(f)
draw = time.time()

def draw_bbox(bbox, results):
    # get the bbox centre (x, y)
    top_left_corner = tuple(bbox[1:3].int())
    # get the bbox width and height
    box_dim = tuple(bbox[3:5].int())

    image = results[int(bbox[0])]
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

# draw the bboxes on the loaded images, inplace
list(map( lambda x: draw_bbox(x, loaded_imgs), output))

# name the output image as 'det_' + image name, in the 'det' directory
det_names = pd.Series(imgs_list).apply( lambda x: '{}/det_{}'.format(args.det, x.split('/')[-1]) )

list(map(cv2.imwrite, det_names, loaded_imgs))
end = time.time()


print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imgs_list)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_loaded - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imgs_list)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()
