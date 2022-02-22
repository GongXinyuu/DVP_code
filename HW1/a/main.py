# -*- coding: utf-8 -*-
# @Date    : 2/21/22
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def main():
    image = Image.open("hw1p4.png").convert("YCbCr")
    ycbcr_image = np.array(image)
    idx = 0  # 0, 1, 2
    plt.imshow(ycbcr_image[:, :, idx:idx+1])
    plt.show()


if __name__ == "__main__":
    main()
