import cv2
import sys
import torch
import argparse

import numpy as np
import torch.nn as nn
import model.model as models
import torchvision

sys.path.append("../")
from parse_config import ConfigParser


def main(config):
    device = torch.device("cpu")
    logger = config.get_logger("kernel extract network loader")

    # initial the transformer network
    transformer = models.kernel_extract_network()
    transformer.encoder = models.kernel_generator(inchannel=3, stride=1, pad=1)
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    logger.debug("loading state_dict to transformer...")
    transformer.load_state_dict(state_dict)
    transformer.decoder = nn.Sequential()
    transformer.to(device)

    logger.debug("successful init transformer ...")

    # # test transformer
    path = f"/home/gauthierli-org/data/data/vehicle-logos-dataset/Source Image Samples/logo thumb6.png"
    org_img = cv2.imread(path).astype(np.float32).transpose((2, 0, 1))
    org_img = torch.from_numpy(org_img).unsqueeze(dim=0).to(device)
    org_img = transformer(org_img)
    # org_img = org_img.squeeze().cpu().data.numpy()[-3:].transpose((1, 2, 0))
    logger.debug("successful transform origin picture ...")
    # cv2.imshow("img", org_img)
    # cv2.waitKey()

    # initial kernel_generator
    kernel_generator = config.init_obj('arch', models)
    kernel_generator.decoder = nn.Sequential()
    kernel_generator.to(device)
    logger.debug("successful initial kernel_generator ...")

    img = "logo_imgs/gray_label/Honda 1_GRAY.jpg"
    img = cv2.imread(img).astype(np.float32).transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(dim=0).to(device)
    img = torchvision.transforms.Resize((243, 243))(img)
    kernel = kernel_generator(img)
    logger.debug("successful generate kernels ...")

    generated_conv = models.load_kernel_network(kernel_tensor=kernel)
    heat_map = generated_conv(org_img)
    show_img = heat_map.squeeze().cpu().data.numpy().transpose((1, 2, 0))
    logger.debug(show_img.shape)
    for i in range(10):
        cv2.imshow("img" + str(i), show_img[:, :, i])
        cv2.waitKey()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="test same transform function")
    args.add_argument('-c', '--config', default=None, type=str, help="config json file path (default)")
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('--log_config', default='../logger/logger_config.json', type=str,
                      help='log_config path (default logger/logger_config.json)')
    config = ConfigParser.from_args(args)
    main(config)
