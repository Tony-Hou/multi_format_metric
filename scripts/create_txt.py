import os
import glob

# root_path = "/home/wuyao/tinghua_project/DDRNet.pytorch/data/wumei_wuzi_all"
# # abspath = os.getcwd()
# # print(abspath)
# # root_path = os.path.abspath("..")
# # print(root_path)
# # ret = abspath.replace(root_path, '.', 1)
# # print(ret)
# data_set = ["leftImg8bit", "gtFine"]
splits = ["train","val"]

root_path = "/home/thu004/DDRNet_new/data/a062"

image_abspath = os.path.join(root_path,"leftImg8bit")
label_abspath = os.path.join(root_path,"gtFine")

for split in splits:
    f = open(os.path.join(root_path,split+".txt"),"w")
    image_list = os.listdir(os.path.join(image_abspath,split))
    # label_list = os.listdir(os.path.join(label_abspath,split))
    for image_name in image_list:
        label_name = image_name[:-4] + ".png"
        image_path = os.path.join("leftImg8bit",split,image_name)
        label_path = os.path.join("gtFine",split,label_name)
        print(image_path)
        print(label_path)
        f.write(image_path+ "\t")
        f.write(label_path+ "\n")


