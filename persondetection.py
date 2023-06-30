import numpy as np
import time
import os
import torch

import cv2
import time
import requests
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple


class DetectorAPI:
    def __init__(self):
       
       cuda = torch.cuda.is_available()
       w = "best.onnx"
       self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
       self.session = ort.InferenceSession(w, providers=self.providers)

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)


    # def processFrame(self, image):
    #     # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
    #     # image_np_expanded = np.expand_dims(image, axis=0)
    #     # Actual detection.
    #     # start_time = time.time()
    #     pred = self.model([image])
    #     # print(image)
    #     # end_time = time.time()

    #     # print("Elapsed Time:", end_time-start_time)
    #     # print(self.image_tensor, image_np_expanded)
    #     labels, cordinates = pred.xyxyn[0][:, -1], pred.xyxyn[0][:,:-1]


    #     n = len(labels)
    #     im_height, im_width, _ = image.shape


        

    #     boxes_list = [None for i in range(n)]
    #     scores = [None for i in range(n)]

    #     for i in range(cordinates.shape[0]):

    #         # row = c[i]
    #         boxes_list[i] = (int(cordinates[i, 0] * im_height),int(cordinates[i, 1]*im_width),int(cordinates[i, 2] * im_height),int(cordinates[i, 3]*im_width))
    #         scores[i] = int(cordinates[i, 4]*100)

    #     return boxes_list, scores, [int(x) for x in labels], int(len(labels))

    def procees_frame(self, img):
        names = ['person']
        colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = self.letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255
        im.shape

        outname = [i.name for i in self.session.get_outputs()]
        # outname

        inname = [i.name for i in self.session.get_inputs()]
        # inname

        inp = {inname[0]:im}

        outputs = self.session.run(outname, inp)[0]

        ori_images = [img.copy()]

        scores = [None for i in range(len(outputs))]
        labels = [0 for i in range(len(outputs))]

        # print(len(labels))

        for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
            image = ori_images[int(batch_id)]
            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            score = round(float(score),3)
            scores[i] = score
            
            name = names[cls_id]
            color = colors[name]
            name += ' '+str(score)
            cv2.rectangle(image,box[:2],box[2:],color,2)
            cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)
        # cv2.imshow("", image)
        # cv2.waitKey(0)
        return image, scores, [int(x) for x in labels]
        

 

# d = DetectorAPI()

# i = cv2.imread("273271,1b9eb00089049cd6.jpg")

# d.procees_frame(i)