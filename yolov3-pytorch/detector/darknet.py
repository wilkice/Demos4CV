"""Implement YOLO-V3 from scratch
pytorch: 1.1.0
python:3.6.8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

from util import predict_transform



def parse_cfg(cfgfile):
    """
    input: cfg file
    return: blocks
    """
    lines = []
    with open(cfgfile, 'r') as f:
        lines = f.read().split('\n')
        lines = [line.strip() for line in lines if len(line)
                 > 0 and not line[0] == '#']

    # get blocks
    block = {}
    block_list = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                block_list.append(block)
                block = {}
            block['type'] = line[1:-1].strip()
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()
    block_list.append(block)
    return block_list



# create modules
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(block_list):
    net_info = block_list[0]
    layer_list = nn.ModuleList()
    prev_filters = 3  # the input channel
    output_filters = []

    for index, x in enumerate(block_list[1:]):
        module = nn.Sequential()

        if x['type'] == 'convolutional':
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            #TODO: why this padding
            # if padding:
            #     pad = (kernel_size -1)//2
            # else:
            #     pad = 0
            pad = padding
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride=stride, padding=pad, bias=bias)
            module.add_module('conv_{}'.format(index), conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batchnorm_{}'.format(index), bn)
            
            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{}'.format(index), activn)
        
        elif x['type'] == 'unsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor= 2, mode='bilinear')
            module.add_module('upsample_{}'.format(index), upsample)

        elif x['type'] == 'route':
            layers = x['layers'].split(',')
            start = int(layers[0])

            try:
                end = int(layers[1])
            except:
                end = 0
            if end > 0:
                end = end-index
            route = EmptyLayer()
            module.add_module('route_{}'.format(index), route)
            if end < 0:
                filters = output_filters[index+start] + output_filters[index+end]
            else:
                filters = output_filters[index + start]

        elif x['type'] =='shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(index), shortcut)

        elif x['type']=='yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1])for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('detection_{}'.format(index), detection)

        layer_list.append(module)
        prev_filters = filters   
        output_filters.append(filters)
    return (net_info, layer_list)

# block_list = parse_cfg('cfg/yolov3.cfg')
# print(create_modules(block_list))


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.block_list = parse_cfg(cfgfile)
        self.net_info, self.layer_list = create_modules(self.block_list)
    
    def forward(self, input, CUDA):
        block_without_net = self.block_list[1:]
        outputs = {}
        write = 0
        detections = 0
        for i, module in enumerate(block_without_net):
            # for every node in block_list without net, total 106
            # module: ['type':'yolo',]
            module_type = module['type']
            if module_type == 'convolutional' or module_type == 'upsample':
                output = self.layer_list[i](input)
            elif module_type == 'route':
                #TODO: not the same
                layers = module['layers']
                layers = layers.split(',')
                if len(layers) == 1:
                    output = outputs[i + int(layers[0].strip())]
                else:
                    start = int(layers[0].strip())
                    end = int(layers[1].strip())
                    map1 = outputs[i + start]
                    map2 = outputs[end]
                    output = torch.cat((map1, map2), 1)
            elif module_type == 'shortcut':
                from_ = int(module['from'])
                output = outputs[i-1] + outputs[i+from_]
            elif module_type == 'yolo':
                #TODO: self.layer_list[i][0]
                anchors = block_without_net[i]['anchors']
                inp_dim = int(self.net_info['height'])
                num_classes = int(module['classes'])

                input = input.data
                output = predict_transform(input, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = output
                    write = 1
                else:
                    detections = torch.cat((detections. output), 1)
            outputs[i] = output
        return detections

def test_input():
    img = cv2.imread('dog-cycle-car.png')
    img = cv2.resize(img, (416,416))
    img = img[:,:,::-1].transpose((2,0,1))
    img = img[np.newaxis,:,:,:] / 255.0
    img = torch.tensor(img).float()
    return img

model = Darknet('cfg/yolov3.cfg')
inp = test_input()
pred = model(inp, torch.cuda.is_available())
print(pred)
