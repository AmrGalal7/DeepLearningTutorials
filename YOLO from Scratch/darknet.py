from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils import *
import cv2

def get_test_input():
    img = cv2.imread('dog-cycle-car.png')
    img = cv2.resize(img, (416, 416))
    # convert 'img' from BGR to RGB, then put it in shape [C, H, W]
    img = img[:, :, ::-1].transpose( (2, 0, 1) )
    # add a new axis for batch. Normalise
    img = img[np.newaxis, :, :, :] / 255.0
    img = torch.from_numpy(img).float()
    img = Variable(img)
    return img


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        # skip the first block: 'net_info'
        modules = self.blocks[1:]
        # a cache for output feature map(s), to be used by 'route' layers
        # key: layer index, value: feature map
        outputs = {}
        # a flag that is set after the first detection
        write = 0
        for i, module in enumerate(modules):
            module_type = module['type']

            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)

            elif module_type == 'route':
                layers_indices = module['layers']
                layers_indices = [ int(l) for l in layers_indices ]
                # if the feature map lies ahead of the current layer 'i', find its relative position to layer 'i'
                if layers_indices[0] > 0:
                    layers_indices[0] -= i
                # if the output is only one feature map, get that layer
                if len(layers_indices) == 1:
                    x = outputs[i + layers_indices[0]]
                # otherwise, concatenate the feature maps along the channel dimension (axis = 1)
                else:
                    # again, if the feature map lies ahead of the current layer 'i', find its relative position to layer 'i'
                    if layers_indices[1] > 0:
                        layers_indices[1] -= i

                    map1 = outputs[i + layers_indices[0]]
                    map2 = outputs[i + layers_indices[1]]

                    x = torch.cat( (map1, map2), 1 )

            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                input_dim = int(self.net_info['height'])
                num_classes = int(module['classes'])

                # transform the prediction/ output feature map shape
                x = x.data
                x = predict_transform(x, input_dim, anchors, num_classes, CUDA = True)

                # if first output
                if not write:
                    detections = x
                    write = 1
                # otherwise, concatenate the output maps along the second dimension
                else:
                    detections = torch.cat( (detections, x), 1 )


            # cache the current layer
            outputs[i] = x


        return detections

    def load_weights(self, weight_file):
        with open(weight_file, 'rb') as f:
            # extract the header of the file, which is 5 int32 values:
            # 1. Major version number
            # 2. Minor version number
            # 3. Subversion number
            # 4, 5. Images seen by the network during training
            header = np.fromfile(f, dtype = np.int32, count = 5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]

            # load the weights, which are stored as float32
            weights = np.fromfile(f, dtype = np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]['type']

            # if the 'module_type' is 'convolutional' load weights,
            # otherwise, continue
            if module_type == 'convolutional':
                model = self.module_list[i]

                try:
                    batch_normalise = int( self.blocks[i + 1]['batch_normalize'] )
                except:
                    batch_normalise = 0

                conv = model[0]

                if(batch_normalise):
                    bn = model[1]

                    # get the number of biases of BatchNorm layers,
                    #  where num of biases = num of weights (filters) = num of means = num of variances
                    num_bn_biases = bn.bias.numel()

                    # load weights:
                    bn_biases = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # reshape the loaded weights into the dimensins of the model weights
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # copy the loaded weights to the model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # load the biases of the convolutional layer
                    num_biases = conv.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr +=  num_biases

                    # reshape the loaded biases into the dimensions of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # copy the loaded biases to the model
                    conv.bias.data.copy_(conv_biases)


                # load the weights of the convolutional layer
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights])
                ptr +=  num_weights

                # reshape the loaded weights into the dimensions of the model weights
                conv_weights = conv_weights.view_as(conv.weight.data)

                # copy the loaded weights to the model
                conv.weight.data.copy_(conv_weights)




class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def parse_cfg(cfgfile):
    '''
    Parameters:
    -----------
                'cfgfile': path to yolo configuration file

    Returns:
    --------
                A list of blocks. Each element of the list is a block in the DNN to be built.
                Each block is a dictionary, with keys:
                    'type': holds the type of block/ module object which could be:
                            'net': containing general info about the DNN,
                            'convolutional',
                            'shorcut': containing description of a skip connection,
                            'route': an output layer/layers concatenated together,
                            'yolo';
                    and other keys depending on 'type' of the block.
                    e.g.: 'mask' and 'anchors' for 'yolo' blocks


    '''

    with open(cfgfile, 'r') as f:
        lines = f.read().split('\n')
        # ignore empty lines and comments
        lines = [x for x in lines if len(x) > 0 and x[0] != '#']
        # remove leading and trailing whitespaces
        lines = [x.lstrip().rstrip() for x in lines]

    block = {}
    blocks = []
    for line in lines:
        if line[0] == '[':
            # if 'block' has been filled, append it to 'blocks'
            if len(block) != 0:
                blocks.append(block)
                block = {}
            # the key 'type' holds the block type. e.g.: convolutional/ shortcut..etc.
            block['type'] = line[1:-1].rstrip()
        else:
            k, v = line.split('=')
            block[k.rstrip()] = v.lstrip()
    blocks.append(block)

    return blocks

def create_modules(blocks):
    '''
    Parameters:
    -----------
                'blocks': a list of info-blocks of the NN to be built.
    Returns:
    --------
                nn.ModuleList
    '''

    # the first block contains some general info. about the DNN
    net_info = blocks[0]
    module_list = nn.ModuleList()
    # number of channels, 3 for RGB
    prev_filters = 3
    # contains the number of channels/ depth of the preceeding layers
    output_filters = []

    for i, block in enumerate(blocks[1:]):
        module = nn.Sequential()

        # Convolutional layer
        if block['type'] == 'convolutional':
            activation = block['activation']
            try:
                batch_normalise = int(block['batch_normalize'])
                bias = False
            except:
                batch_normalise = 0
                bias = True
            filters = int(block['filters'])
            padding = int(block['pad'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            # for same padding
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module('conv_{0}'.format(i), conv)

            if batch_normalise:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(i), bn)

            # Note: add more conditions if there're more options on the activation type
            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module('leaky_{0}'.format(i), activn)


        # Upsampling layer
        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            # upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear')
            upsample = nn.Upsample(scale_factor = stride, mode = 'nearest')
            module.add_module('upsample_{0}'.format(i), upsample)

        # Route
        # defines which output feature map(s) is the output of YOLO.
        # if the output is only one feature map, its index is denoted by 'first_output_index';
        # however, if the there're more than one feature map, the feature maps, designated by
        # 'first_output_index' and 'last_output_index', are concatenated together.
        elif block['type'] == 'route':
            block['layers'] = block['layers'].split(',')
            first_output_index = int(block['layers'][0])
            try:
                last_output_index = int(block['layers'][1])
            except:
                last_output_index = 0

            # positive indices
            if first_output_index > 0:
                first_output_index -= i
            if last_output_index > 0:
                last_output_index -= i
            route = EmptyLayer()
            module.add_module('route_{0}'.format(i), route)

            # get the number of channels needed for a possibly upcoming convolutional layer
            if last_output_index < 0:
                # in case of concatenated output
                filters = output_filters[i + first_output_index] + output_filters[i + last_output_index]
            else:
                filters = output_filters[i + first_output_index]

        # Skip connection
        elif block['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shorcut_{0}'.format(i), shortcut)

        # Yolo
        elif block['type'] == 'yolo':
            mask = block['mask'].split(',')
            mask = [ int(x) for x in mask ]

            anchors = block['anchors'].split(',')
            anchors = [ int(x) for x in anchors ]
            anchors = [ (anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2) ]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('Detection_{0}'.format(i), detection)


        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)








# blocks = parse_cfg('./config/yolov3.cfg')
# print(create_modules(blocks))

# CUDA = torch.cuda.is_available()
# model = Darknet('./config/yolov3.cfg')
# input_img = get_test_input()
# if CUDA:
#     model = model.cuda()
#     input_img = input_img.to('cuda')
#
# model.load_weights("yolov3.weights")
# pred = model(input_img, CUDA)
# print(pred.size())
