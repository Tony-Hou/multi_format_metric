# ------------------------------------------------------------------------------
#
# Written by Alex
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel, speed_test
import onnxruntime as ort
from tqdm import tqdm
from utils.utils import Map16, Vedio
import imageio
import cv2

vedioCap = Vedio('./output/test_out.mp4')
map16 = Map16(vedioCap)


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default=" experiments/wumei_wuzi/wuzi23_slim.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


class onnx_net(nn.Module):
    def __init__(self, model):
        super(onnx_net, self).__init__()
        self.backone = model

    def forward(self, x):
        x1, x2 = self.backone(x)
        # y = F.interpolate(x1, size=(512,512), mode='bilinear')
        y = F.interpolate(x1, size=(512, 512), mode='nearest')
        # y = F.softmax(y, dim=1)
        # y = torch.argmax(y, dim=1)

        return y

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))


    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_samples=None,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    output_path = "/home/gs/disk1/project/shangzong/DDRNet/IR_models/shangchao_shangzong/nearest/v002/wuzi_slim.onnx"
    ort_session = ort.InferenceSession(output_path.__str__())
    ort_inputs = ort_session.get_inputs()[0].name
    for _, batch in enumerate(tqdm(testloader)):
        image, size, name = batch
        image = F.interpolate(
            image, size=(512, 512),
            mode='bilinear')

        size = size[0]
        pred = ort_session.run([], {ort_inputs: to_numpy(image)})
        pred = torch.from_numpy(np.squeeze(np.array(pred), axis=0))
        if pred.shape[-2] != size[0] or pred.shape[-1] != size[1]:
            pred = F.interpolate(
                pred, size=(480, 640),
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )

        # _, pred = torch.max(pred, dim=1)
        # pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)
        # print(pred)
        # pred_index = pred[:] == 3
        # pred[pred_index] = 1
        # imageio.imsave(os.path.join(sv_dir, name[0] + '.png'), pred)
        # process ori image
        image = image.squeeze(0)
        image = image.numpy().transpose((1,2,0))
        image *= [0.229, 0.224, 0.225]
        image += [0.485, 0.456, 0.406]
        image *= 255.0
        image = image.astype(np.uint8)
        _, pred = torch.max(pred, dim=1)
        pred = pred.squeeze(0).cpu().numpy()
        map16.visualize_result(image, pred,
                            "/home/gs/disk1/project/sewage/ddrnet/output/wumei_wuzi_all/wuzi23_slim/a062_test_result/", name[0]+'.png')

if __name__ == '__main__':
    main()
