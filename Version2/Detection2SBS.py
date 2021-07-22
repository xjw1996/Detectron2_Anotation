import numpy as np
import cv2 as cv
from PIL import Image
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import random
#from google.colab.patches import cv2_imshow
from cv2.xfeatures2d import matchGMS
from enum import Enum
import detectron2
import torch
from detectron2.utils.logger import setup_logger
setup_logger()

import cv2
import os
import time



from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


from detectron2 import model_zoo


class SegmentationDetector():
    def DetectorSingleImage(self,image):
        cfg = get_cfg()
        cfg.merge_from_file("/home/chen/detectron3/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        cfg.MODEL.WEIGHTS = "/home/chen/detectron3/detectron2/model/model_final_cafdb1.pkl"
        predictor = DefaultPredictor(cfg)
        output = predictor(image)
        # print(output)
        mask = output["instances"].pred_masks.cpu().numpy()
        # print(len(mask))
        panoptic_seg, segments_info = predictor(image)["panoptic_seg"]
        # print(segments_info)

        # print(panoptic_seg)
        # print(panoptic_seg[0])
        # print("len",len(panoptic_seg))
        # print("len",len(panoptic_seg[0]))
        # print("len",len(image))
        # print("len",len(image[0]))

        meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
        
        # get()方法语法： dict.get(key, de?fault=None)
        # for data in range(len(segments_info)):

        #     if(segments_info[data].get("isthing") == True):
        #         print(meta.thing_classes[segments_info[data].get("category_id")])
        #         print(segments_info[data].get("category_id"))
        #         print("----------------------------------------------------------------------")
        #     else:
        #         print(meta.stuff_classes[segments_info[data].get("category_id")])
        #         print(segments_info[data].get("category_id"))
        #         print("----------------------------------------------------------------------")
        IsNotThing = []
        for x in range(len(segments_info)):
                if(segments_info[x].get("isthing") == True):
                    pass
                else:
                    IsNotThing.append(segments_info[x].get("id"))

        # print(IsNotThing)

                    
        

        # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        # cv2.imshow('image', v.get_image()[:,:,::-1])
        # cv2.waitKey(0)

        temp_mask_fill = np.zeros((len(image), len(image[0])))
        # 处理一章图片需要2分40秒  血受
        # # temp_mask_fill=temp_mask_fill.tolist()
        # for x in range(len(panoptic_seg)):
        #     for y in range(len(panoptic_seg[0])):   
        #         if(panoptic_seg[x][y] in IsNotThing):
        #             temp_mask_fill[x][y] = int(0)
        #             # print(temp_mask_fill[x][y])
        #         else:
        #             temp_mask_fill[x][y] = 255
     
        panoptic_seg = panoptic_seg.data.cpu().numpy()


        # 遍历一个矩阵
        # for i in np.nditer(panoptic_seg, order='C'):
        #     print(i)


        for x,y in np.nditer([panoptic_seg,temp_mask_fill],op_flags = ['readwrite'],order='C'):
            if(x in IsNotThing):
                y[...] = 0
            else:
                y[...] = 255
                
        
        return temp_mask_fill


class MoveingObjectOriginalColor():

    def FindOriginalColor(self,OriginImage,MoveingImage):
        #数据由原来的0.  变为0   否则找不到contours
        MoveingImage = MoveingImage.astype(np.uint8)
        contours, hierarchy = cv2.findContours( MoveingImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = contours[0]
        max_area = 0
        for contour_one in contours:
            area = cv2.contourArea(contour_one)
            if (area > max_area):
                max_contour = contour_one
                max_area = area

        # print("I am max_contour",max_contour)
        # print("I am max_area",max_area)

        mask = np.zeros((len(MoveingImage), len(MoveingImage[0]),3))
        
        cv2.fillConvexPoly(mask, max_contour, (255,255,255))

        #给MoveingImage的白色地方加上圆图的RGB
        # print(type(OriginImage[0][0]))
        
        for x,y in np.nditer([mask,OriginImage],op_flags = ['readwrite'],order='C'):
            if(x == 0.0):
                y[...]=x
            else:
                pass
        
        return OriginImage
        # cv2.imshow('image', OriginImage)
        # cv2.waitKey(0)





class InstancePerson():

    def instanceperson(self,image):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
        cfg.MODEL.WEIGHTS = "/home/chen/detectron3/detectron2/model/model_final_f10217.pkl"
        cfg.MODEL.DEVICE ="cuda"

        predictor = DefaultPredictor(cfg)


        output = predictor(image)
        v= Visualizer(image[:,:,::-1],
                scale=0.8,
                metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                instance_mode=ColorMode.IMAGE
                )
        # v=v.draw_instance_predictions(output['instances'][output['instances'].pred_classes == 0].to("cpu"))
        # cv2.imshow('image', v.get_image()[:,:,::-1])
        # cv2.waitKey(0)
        instances_list=[]
        index_list=[]
        #   ------- 输出istance 检测出来的物体的号码 --------------
        # print("the number of class",output["instances"].pred_classes)
        #   ------- 输出istance 检测出来的物体 --------------
        for data in output["instances"].pred_classes:
            num = data.item()
            # print("我是num",num)
            instances_list.append(num)
            # print("the class of person",MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[num])

        # print(instances_list)

        #计算instance人的数量有多少
        PersonNumTimes=0
        for nums in instances_list:
            if(nums == 0):
                index_list.append(PersonNumTimes)
            PersonNumTimes = PersonNumTimes + 1

        image_thresholding = np.zeros((len(image), len(image[0])))
        #判断instance person的数量是否为1
        # if(len(index_list)==1):
        mask_persion = output["instances"].pred_masks.cpu().numpy()[0] 
        temp_mask_persion = mask_persion.astype(int)
        for x in np.nditer(temp_mask_persion,op_flags = ['readwrite'],order='C'):
            if(x == 0):
                pass
            else:
                x[...] = 255

        temp_mask_persion = temp_mask_persion.astype('uint8')
        # cv2.imshow('image', temp_mask_persion)
        # cv2.waitKey(0)
        # print(temp_mask_persion)
        temp_mask_persion = cv2.dilate(temp_mask_persion, np.ones((30, 30)))


        # cv2.imshow('image', temp_mask_persion)
        # cv2.waitKey(0)

        for x in range(len(temp_mask_persion)):
            for y in range(len(temp_mask_persion[0])):
                if (temp_mask_persion[x][y] == 0):
                    pass
                else:
                    image[x][y] = [0,0,0]


        for x in range(len(temp_mask_persion)):
            for y in range(len(temp_mask_persion[0])):
                if (image[x][y][0] == 0 ):
                    image_thresholding[x][y] = int(0)
                else:
                    image_thresholding[x][y] = int(255)
        # else:
        #     for person_num in range(3):
        #         mask_persion = output["instances"].pred_masks.cpu().numpy()[person_num] 
        #         temp_mask_persion = mask_persion.astype(int)
        #         for x in np.nditer(temp_mask_persion,op_flags = ['readwrite'],order='C'):
        #             if(x == 0):
        #                 pass
        #             else:
        #                 x[...] = 255
                
        #         temp_mask_persion = temp_mask_persion.astype('uint8')
        # print("我是数组索引",index_list)

        
        cv2.imshow('image', image_thresholding)
        cv2.waitKey(0)
        

        return image_thresholding


class DrawTheBoundingbox():
    def draw(self,image,threshold):
        threshold = threshold.astype(np.uint8)
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 小さい輪郭は誤検出として削除する
        # contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))
        print("index of contours",len(contours))
        max_contour = contours[0]
        # print(max_contour)
        max_area = 0
        for contour_one in contours:
            area = cv2.contourArea(contour_one)
            if (area > max_area):
                max_contour = contour_one
                max_area = area
        # im = cv2.drawContours(im, [max_contour], 0, (0, 255, 0), 3)
        # 輪郭を描画する。

        rect = max_contour
        x, y, w, h = cv2.boundingRect(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 10)

          
        cv2.drawContours(image, max_contour, -1, color=(0, 0, 255), thickness=2)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        return image,x,y


class SaveDataSet(): 
    def saveimages(self, OriginalColor, Box, dir_path,label_index,x,y,img_width,img_height,current_picture):
        timestamp = int(round(time.time() * 1000))
        cv2.imwrite(dir_path + 'images/' + str(timestamp) + '.jpg', OriginalColor)
        cv2.imwrite(dir_path + 'annotations/' + str(timestamp) + '.jpg', Box)


        with open(dir_path + 'labels/' + str(timestamp) + '.txt', mode='w+') as f:
            f.write(
                label_index + ' ' +
                str(x) + ' ' +
                str(y) + ' ' +
                str(img_width) + ' ' +
                str(img_height)
                )

        with open(dir_path + 'total.txt', mode='a+') as f:
            f.write(
                current_picture + "\n"
            )

class DrawingType(Enum):
        ONLY_LINES = 1
        LINES_AND_POINTS = 2
        COLOR_CODED_POINTS_X = 3
        COLOR_CODED_POINTS_Y = 4
        COLOR_CODED_POINTS_XpY = 5


def draw_matches(src1, src2, kp1, kp2, matches, drawing_type):
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]

    if drawing_type == DrawingType.ONLY_LINES:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255))

    elif drawing_type == DrawingType.LINES_AND_POINTS:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
            cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)

    elif drawing_type == DrawingType.COLOR_CODED_POINTS_X or drawing_type == DrawingType.COLOR_CODED_POINTS_Y or drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
        _1_255 = np.expand_dims(np.array(range(0, 256), dtype='uint8'), 1)
        _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))

            if drawing_type == DrawingType.COLOR_CODED_POINTS_X:
                colormap_idx = int(left[0] * 256. / src1.shape[1])  # x-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_Y:
                colormap_idx = int(left[1] * 256. / src1.shape[0])  # y-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
                colormap_idx = int((left[0] - src1.shape[1]*.5 + left[1] - src1.shape[0]*.5) * 256. / (src1.shape[0]*.5 + src1.shape[1]*.5))  # manhattan gradient

            color = tuple(map(int, _colormap[colormap_idx, 0, :]))
            cv2.circle(output, tuple(map(int, left)), 1, color, 2)
            cv2.circle(output, tuple(map(int, right)), 1, color, 2)
    return output

if __name__ == "__main__":
    path = r'/home/chen/detectron3/detectron2/images/test3/output1'

    dirs = os.listdir(path)

    label_inde = "00"


    # -------------------------  test single picture -----------------------

    # current_picture = "/home/chen/detectron3/detectron2/images/test3/output1/"+"37.jpg"
    # img1 = cv2.imread("/home/chen/图片/Screenshot from 2021-07-21 19-08-04.png")
    # img2 = cv2.imread(current_picture)
    # im = cv.imread(current_picture)
    # im2 = cv.imread(current_picture)
    # ori = cv.imread(current_picture)

    # img_shape=im.shape
    # img_width = img_shape[1]
    # img_height = img_shape[0]

    # print("----------------------Now"+current_picture+"-------------------------is processing")
    # dir_path = "/home/chen/detectron3/detectron2/images/dataset3/"
    # GotoPredictor = SegmentationDetector()
    # MoveingObject = GotoPredictor.DetectorSingleImage(im)
    # # print(MoveingObject)

    # GotoOriginalColor = MoveingObjectOriginalColor()
    # OriginalColor = GotoOriginalColor.FindOriginalColor(im,MoveingObject)
  
    # Persion=InstancePerson()
    # GetPersion = Persion.instanceperson(OriginalColor)

    # DrawBox = DrawTheBoundingbox()
    # Box = DrawBox.draw(im2,GetPersion)
   

    # output = draw_matches(img1, OriginalColor, kp1, kp2, matches_gms, DrawingType.ONLY_LINES)
    # cv2.imwrite(dir_path + 'gms_feacher/' + str(timestamp) + '.jpg', output)
    # cv2.imshow('image', output)
    # cv2.waitKey(0)





    # -------------------------  test dataset -----------------------

    for image in dirs:
        current_picture = "/home/chen/detectron3/detectron2/images/test3/output1/"+image
        img1 = cv2.imread("/home/chen/图片/Screenshot from 2021-07-21 19-08-04.png")
        img2 = cv2.imread(current_picture)
        im = cv.imread(current_picture)
        im2 = cv.imread(current_picture)
        ori = cv.imread(current_picture)

        img_shape=im.shape
        img_width = img_shape[1]
        img_height = img_shape[0]

        print("----------------------Now"+current_picture+"-------------------------is processing")
        dir_path = "/home/chen/detectron3/detectron2/images/dataset3/"
        GotoPredictor = SegmentationDetector()
        MoveingObject = GotoPredictor.DetectorSingleImage(im)
        # print(MoveingObject)

        GotoOriginalColor = MoveingObjectOriginalColor()
        OriginalColor = GotoOriginalColor.FindOriginalColor(im,MoveingObject)
        orb = cv2.ORB_create(10000)
        orb.setFastThreshold(0)

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(OriginalColor, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches_all = matcher.match(des1, des2)

        matches_gms = matchGMS(img1.shape[:2], OriginalColor.shape[:2], kp1, kp2, matches_all, withScale=False, withRotation=False, thresholdFactor=6)
        timestamp = int(round(time.time() * 1000))
        output = draw_matches(img1, OriginalColor, kp1, kp2, matches_gms, DrawingType.ONLY_LINES)
        cv2.imwrite(dir_path + 'gms_feacher/' + str(timestamp) + '.jpg', output)
        # cv2.imshow('image', output)
        # cv2.waitKey(0)

        # -------output number of gms number---------------
        # print("output of matches_gms's number",len(matches_gms))
        if(len(matches_gms)<10):
            pass
        else:
            Persion=InstancePerson()
            GetPersion = Persion.instanceperson(OriginalColor)

            DrawBox = DrawTheBoundingbox()
            Box = DrawBox.draw(im2,GetPersion)
            x=Box[1]+(img_width/2)
            y=Box[2]+(img_height/2)
            SaveImage = SaveDataSet()
            SaveImage.saveimages(ori, Box[0], dir_path, label_inde,Box[1], Box[2],img_width,img_height,current_picture)


    # print('Found', len(matches_gms), 'matches')
    # cv2.imshow('image', img1)
    # cv2.waitKey(0)
