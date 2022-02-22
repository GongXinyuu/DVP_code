# -*- coding: utf-8 -*-
# @Date    : 2/21/22
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -0.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


def main(sample_rate=2, interpolation=InterpolationMode.BILINEAR):
    image = Image.open("hw1p4.png").convert("YCbCr")
    ycbcr_image = np.array(image)

    # idx = 1  # 1, 2
    cb_channel = ycbcr_image[:, :, 1]
    cr_channel = ycbcr_image[:, :, 2]

    resampled_cb_channel = cb_channel[::sample_rate, ::sample_rate]
    resampled_cr_channel = cr_channel[::sample_rate, ::sample_rate]

    resampled_cb_channel = (
        F.resize(
            torch.from_numpy(resampled_cb_channel).unsqueeze(0),
            size=ycbcr_image.shape[:2],
            interpolation=interpolation,
        )
        .squeeze(0)
        .numpy()
    )
    resampled_cr_channel = (
        F.resize(
            torch.from_numpy(resampled_cr_channel).unsqueeze(0),
            size=ycbcr_image.shape[:2],
            interpolation=interpolation,
        )
        .squeeze(0)
        .numpy()
    )

    resampled_image = np.stack(
        [ycbcr_image[:, :, 0], resampled_cb_channel, resampled_cr_channel], axis=-1
    )
    plt.imshow(ycbcr2rgb(resampled_image))
    plt.show()


if __name__ == "__main__":
    sampling_rate = 16
    interp_method = InterpolationMode.NEAREST  # InterpolationMode.BILINEAR
    main(sampling_rate, interpolation=interp_method)
