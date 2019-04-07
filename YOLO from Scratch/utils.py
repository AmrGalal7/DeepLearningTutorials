from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

CUDA = torch.cuda.is_available()


def letterbox_image(img, input_dim):
    '''
        resizes the 'img' while keeping the aspect ratio unchanged, and padding
        the left out areas with colour (128, 128, 128)
    '''
    img_h, img_w = img.shape[0], img.shape[1]
    w, h = input_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
    canvas = np.full( (h, w, 3), 128 )
    # insert the resized image in the middle of the (128, 128, 128)-coloured 'canvas'
    canvas[ (h-new_h) // 2 : (h-new_h) // 2 + new_h, (w-new_w) // 2 : (w-new_w) // 2 + new_w, : ] = resized_img

    return canvas

def prep_image(img, input_dim):
    '''
        converts opencv's BGR numpy array into an RGB pytorch tensor of format (batches, channels, height, width)

        Returns:
        --------
            torch variable for the image, normalised.
    '''
    img = letterbox_image(img, (input_dim, input_dim))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    # normalise by diving by 255
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def load_classes(namesfile):
    '''
        returns a list of classes in 'namesfile's
    '''
    with open(namesfile, "r") as f:
        names = f.read().split("\n")[:-1]
    return names

def predict_transform(prediction, input_dim, anchors, num_classes, CUDA = True):
    '''
        transforms the the prediction (output feature map) from size [ num_anchors * bbox_attrs, feature_map_height, feature_map_width],
        to size [ feature_map_height * feature_map_width * num_anchors, bbox_attrs], so that each row becomes the attributes of a certain bounding box.

        Note: bbox_attrs is 5 + 'num_classes'; 5 accounts for box confidence (to), predicted centre (tx, ty), and predicted anchor dimensions (th, tw)
        Note2: this description is for each map per batch (i.e. 'prediction' is a batch of feature maps)

        Parameters:
        -----------
                prediction: the detection layer (output feature map)
                input_dim: height (or width) of input images
                anchors: a list of the predefined anchor boxes, is a tuple of (height, width) w.r.t. the input image dimension
                num_classes: number of classes the DNN was trained to classify

        Returns:
        --------
                prediction: the output feature map resized into [ feature_map_height * feature_map_width * num_anchors, bbox_attrs]

    '''

    batch_size = prediction.size(0)
    # size of the feature map
    grid_size =  prediction.size(2)
    stride = input_dim // grid_size
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # prediction = prediction.to(torch.device("cuda"))

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    # swap the bounding boxes attributes with the grid cells, and save the tensor in a contiguous block in the memory (required by the loss)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    # scale the anchors dimensions w.r.t. the current feaure map dimensions, instead of being relative to the input image
    anchors = [ (anchor[0] / stride, anchor[1] / stride) for anchor in anchors ]

    # apply the sigmoid function to the predicted box centre (tx, ty) and box confidence (to), respectively
    prediction[:, :, 0] = torch.sigmoid( prediction[:, :, 0] )
    prediction[:, :, 1] = torch.sigmoid( prediction[:, :, 1] )
    prediction[:, :, 4] = torch.sigmoid( prediction[:, :, 4] )

    # add the centre offsets to the centre coordinates
    grid = np.arange(grid_size)
    xv, yv = np.meshgrid(grid, grid)
    # reshape x_offset into a 2D tensor of size [grid_size * grid_size, 1]
    x_offset = torch.FloatTensor(xv).view(-1, 1)
    y_offset = torch.FloatTensor(yv).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
        prediction = prediction.cuda()


    # concatenate the x and y offsets and repeat them for each anchor box:
    # 1. concatenate x and y offsets into a [grid_size * grid_size, 2] tensor
    # 2. repeat the offset 'num_anchors' times along the columns -> [grid_size * grid_size, 2 * 'num_anchors']
    # 3. flatten the columns into only 2 -> [grid_size * grid_size * 'num_anchors', 2]
    # 4. turn the tensor into a 3D one, for the next addition step -> [1, grid_size * grid_size * 'num_anchors', 2]
    x_y_offset = torch.cat( (x_offset, y_offset), 1 ).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # get the anchors' default size (height, width)
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    # repeat and put the anchors in size [1, grid_size*grid_size, 2]
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    # apply exp to the predicted anchor dimentions (th, tw) and multiply by anchors default sizes
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # apply sigmoid to the classes scores
    prediction[:, :, 5:5 + num_classes] = torch.sigmoid(prediction[:, :, 5:5 + num_classes])

    # resize the feature map predictions to the size of the input image
    prediction[:, :, :4] *= stride


    return prediction


def unique(tensor):
    '''
    gets the unique values of a tensor

    Paramaters:
    -----------
            tensor: a 1D tensor

    Returns:
    --------
            a 1D tensor of the unique values from the given 'tensor'
    '''

    # to convert the tensor into a numpy array, it first has to be moved to cpu
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def bbox_iou(box1, box2):
    '''
    returns the IOU between 'box1' and the other bboxes in 'box2'

    Parameters:
    -----------
                box1: predictions of a certain bbox, of size [1, 7]
                box2: predictions of several bboxes, of size [box2.size(0), 7]

    Returns:
    --------
            (inter_area / union_area), of size [box2.size(0), ]
    '''
    # get the bboxes coordinates:
    b1_top_left_corner_x, b1_top_left_corner_y, b1_bottom_right_corner_x, b1_bottom_right_corner_y = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_top_left_corner_x, b2_top_left_corner_y, b2_bottom_right_corner_x, b2_bottom_right_corner_y = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the coordinates of the region (rectangle) of intersection
    inter_top_left_corner_x = torch.max(b1_top_left_corner_x, b2_top_left_corner_x)
    inter_top_left_corner_y = torch.max(b1_top_left_corner_y, b2_top_left_corner_y)
    inter_bottom_right_corner_x = torch.min(b1_bottom_right_corner_x, b2_bottom_right_corner_x)
    inter_bottom_right_corner_y = torch.min(b1_bottom_right_corner_y, b2_bottom_right_corner_y)

    # calculate the intersection area
    inter_area = torch.clamp(inter_bottom_right_corner_x - inter_top_left_corner_x, min = 0) * \
                 torch.clamp(inter_bottom_right_corner_y - inter_top_left_corner_y, min = 0)

    # calculate the union area: box1 area + box2 area - 'inter_area'
    union_area = (b1_bottom_right_corner_x - b1_top_left_corner_x) * (b1_bottom_right_corner_y - b1_top_left_corner_y) + \
                 (b2_bottom_right_corner_x - b2_top_left_corner_x) * (b2_bottom_right_corner_y - b2_top_left_corner_y) - \
                 inter_area

    return (inter_area / union_area)

def write_results(prediction, box_confidence, num_classes, nms_confidence = 0.4):
    '''
    Paramaters:
    -----------
            prediction: the detection in size [batch, total_num_bbox, 5 + num_classes], where
                        total_num_bbox = sum(grid_size_i * grid_size_i * 3) for all i detection layers
            box_confidence: objectness score threshold
            num_classes,
            nms_confidence: threshold for non-maximum suppression of IOUs

    Returns:
    --------
            output: a tensor holding the predictions of the whole batch, of size [x, 8],
                    where x: number of bboxes after IOU and NNM thresholding,
                          8: image index + the usual 7 predictions of a bbox


    '''

    # IOU thresholding:
    # get the indices of the bbox with objectness score > 'box_confidence' threshold -> [batch, total_num_bbox]
    # add the indices in a new axis -> [batch, total_num_bbox, 1]
    box_confidence_mask = (prediction[:, :, 4] > box_confidence).float().unsqueeze(2)
    # zero out the bboxes w/ objectness score < 'box_confidence' threshold
    prediction = prediction * box_confidence_mask

    # try:
    #     ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    # except:
    #     return 0

    # transform the bbox attributes ( centre_x, centre_y, width, height ) into:
    # ( top_left_corner_x, top_left_corner_y, bottom_right_corner_x, bottom_right_corner_y )
    box_corners = prediction.new(prediction.shape)
    box_corners[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corners[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corners[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corners[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corners[:, :, :4]

    # loop over the batch images to apply confidence thresholding and NMS,
    # since detections by an image is independent of the other images in the batch
    batch_size = prediction.size(0)

    # indicates whether detections have been made
    # detections made are stored in 'outputs' tensor in 'Darknet'
    write = False

    for ind in range(batch_size):
        img_pred = prediction[ind]
        # get the values and indices of the classes having max conditional class probability
        max_cond_prob_value, max_cond_prob_i = torch.max(img_pred[:, 5: 5 + num_classes], 1)
        max_cond_prob_value = max_cond_prob_value.float().unsqueeze(1)
        max_cond_prob_i = max_cond_prob_i.float().unsqueeze(1)

        # concatenate the bbox predictions, value of the max conditional class probability,
        # and the index of that class into one tensor 'img_pred'
        sequence = (img_pred[:, :5], max_cond_prob_value, max_cond_prob_i)
        img_pred = torch.cat(sequence, 1)

        # get the indices of the bboxes with objectness score > 'box_confidence' threshold,
        # those below the threshold has been set to zero at the very first few lines of the this function.
        # non_zero_indices is of size [total_num_bbox, 1]
        non_zero_indices = torch.nonzero(img_pred[:, 4])


        # if there's a detection, get the predictions
        # '7' accounts for the bbox coordinates (4) + objectness score +
        # value and index of the class with max conditional class probability (2)
        # view(-1, 7)?
        # try:
        img_pred_ = img_pred[non_zero_indices.squeeze(), :].view(-1, 7)
        # except:
        #     continue

        try:
            # get the detected classes if there are any, knowing that the index of the class with
            # max conditional class probability is the last element in each row
            # view(-1, 7)?
            img_classes = unique(img_pred_[:, -1])
        except:
            continue


        # For PyTorch 0.4 compatibility
        # Since the above code with not raise exception for no detection
        # as scalars are supported in PyTorch 0.4
        if img_pred_.shape[0] == 0:
            continue

        # Non-maximum suppression:
        # loop over the detected classes
        for c in img_classes:
            # get the bbox(es) that predict the class 'c'
            class_mask = img_pred_ * (img_pred_[:, -1] == c).float().unsqueeze(1)
            # find the nonzero indices (along any column)?, and apply the mask
            class_mask_indices = torch.nonzero(class_mask[:, -2]).squeeze()
            img_pred_class = img_pred_[class_mask_indices].view(-1, 7)
            # sort the bbox(es) descendingly according to their objectness confidence,
            # then sort the bbox(es) accordingly
            obj_conf_sort_indices = torch.sort(img_pred_class[:, 4], descending = True)[1]
            img_pred_class = img_pred_class[obj_conf_sort_indices]
            # number of bbox(es) predicting class 'c'
            num_detections = img_pred_class.size(0)

            for i in range(num_detections):
                # loop over bboxes after bbox(i), and get their IOUs w.r.t. the current bbox(i)
                try:
                    # 'ious' can have different lengths
                    ious = bbox_iou(img_pred_class[i].unsqueeze(0), img_pred_class[i+1:])

                # img_pred_class[i+1:] returns an empty slice ( no detections other than bbox(i) ), break.
                except ValueError:
                    break
                # Non-maximum suppression can no further removes bboxes if:
                # 'i' out of bounds, due to the removal of entries from 'img_pred_class', break
                except IndexError:
                    break

                # keep only the detections having IOUs < 'nms_confidence' threshold
                img_pred_class[i+1:] *= (ious < nms_confidence).float().unsqueeze(1)
                # remove the zero entries (bboxes)
                non_zero_ious = torch.nonzero(img_pred_class[:, 4]).squeeze()
                img_pred_class = img_pred_class[non_zero_ious].view(-1, 7)

            # add the image index, as a column, to our predictions
            image_index = img_pred_class.new(img_pred_class.size(0), 1).fill_(ind)
            sequence = (image_index, img_pred_class)

            # if it's the first image
            if not write:
                output = torch.cat(sequence, 1)
                write = True
            else:
                out = torch.cat(sequence, 1)
                output = torch.cat((output, out))


    # check whether there's at least one prediction among the batch
    try:
        return output
    except:
        return 0
