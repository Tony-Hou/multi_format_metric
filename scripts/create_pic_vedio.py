# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np


mask_path = "/home/wuyao/tinghua_project/FCHarDNet_wuzi/a062/gtFine_3/val"
raw_image_path = "/home/wuyao/tinghua_project/FCHarDNet_wuzi/test_out_rgb_wuzi_a062_20210318"
image_mask_path = "/home/wuyao/tinghua_project/DDRNet/output/wumei_wuzi_all/wuzi23_slim/a062_test_result"

vidio_dir = "/home/wuyao/tinghua_project/DDRNet/video"


# 图片合成视频
def picvideo(raw_image_path , image_mask_path,size):
    # path = r'C:\Users\Administrator\Desktop\1\huaixiao\\'#文件路径
    filelist = os.listdir(raw_image_path)  # 获取该目录下的所有文件名
    #ilelist.sort();
    # filelist.sort(key = lambda x: int(x.split("_")[-1][:-4]))
    # print(filelist)

    '''
    fps:
    帧率：1秒钟有n张图片写进去[控制一张图片停留5秒钟，那就是帧率为1，重复播放这张图片5次] 
    如果文件夹下有50张 534*300的图片，这里设置1秒钟播放5张，那么这个视频的时长就是10秒
    '''
    fps = 1
    # size = (591,705) #图片的分辨率片
    #image_path1.strip().split("/")[-1]

    video_name = raw_image_path.strip().split("/")[-3] +"_0318.mp4";
    video_path = os.path.join(vidio_dir,video_name)
    print(video_path)
    #fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

    video = cv2.VideoWriter(video_path, fourcc, fps, size)

    for item in filelist:
        if item.endswith('.jpg'):  # 判断图片后缀是否是.jpg
            #img = np.zeros((1200,600,3))

            item1 = mask_path + '/' + item[:-4]+".png"
            img1 = cv2.imread(item1)  # 使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
            img1 = cv2.resize(img1, (640, 480))


            item2 = raw_image_path + '/' + item
            img2 = cv2.imread(item2)
            img2 = cv2.resize(img2, (640,480))
            #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            item3 = image_mask_path + '/' +item
            img3 = cv2.imread(item3)
            img3 = cv2.resize(img3, (640, 480))

            img = np.concatenate((img1,img2,img3),axis=1)
            # img = np.concatenate((img,),axis=1)

            # img1 = np.array(img1)
            # img2 = np.array(img2)

            #img[44:556, 58:570,  :] += img1
            #img[44:556, 628:1140,:] += img2
            #print(img.shape)
            #cv2.imwrite("/home/wuyao/pytorch_openvino_output/"+item,img)
            video.write(img)  # 把图片写进视频

    video.release()  # 释放

picvideo(raw_image_path,image_mask_path,(1920,480))
# img = cv2.imread("/home/wuyao/pytorch_openvino_output/alldirtywater_2.jpg")
# print(img.shape)