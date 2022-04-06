import os
import sys
import time
import operator
import cv2
import numpy as np
from numpy import linalg as LA
from PIL import Image
from sklearn import svm
import joblib  # save / load model

"""
# Download INRIAPerson dataset:
$ wget ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar
$ tar -xf INRIAPerson.tar

Driver
https://drive.google.com/open?id=1HYR2wfXRkj9Jg979c9CwFeDVL574hR9U
"""

MODEL_PATH = 'models/model_hog_person.joblib'
IMG_PATH = 'woman.jpg'


def hog(img_gray, cell_size=8, block_size=2, bins=9):
    img = img_gray
    h, w = img.shape  # 128, 64

    # gradient
    xkernel = np.array([[-1, 0, 1]])
    ykernel = np.array([[-1], [0], [1]])
    dx = cv2.filter2D(img, cv2.CV_32F, xkernel)
    dy = cv2.filter2D(img, cv2.CV_32F, ykernel)

    # histogram
    magnitude = np.sqrt(np.square(dx) + np.square(dy))
    orientation = np.arctan(np.divide(dy, dx + 0.00001))  # radian
    orientation = np.degrees(orientation)  # -90 -> 90
    orientation += 90  # 0 -> 180

    num_cell_x = w // cell_size  # 8
    num_cell_y = h // cell_size  # 16
    hist_tensor = np.zeros([num_cell_y, num_cell_x, bins])  # 16 x 8 x 9
    for cx in range(num_cell_x):
        for cy in range(num_cell_y):
            ori = orientation[cy * cell_size:cy * cell_size + cell_size, cx * cell_size:cx * cell_size + cell_size]
            mag = magnitude[cy * cell_size:cy * cell_size + cell_size, cx * cell_size:cx * cell_size + cell_size]
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
            hist, _ = np.histogram(ori, bins=bins, range=(0, 180), weights=mag)  # 1-D vector, 9 elements
            hist_tensor[cy, cx, :] = hist
        pass
    pass

    # normalization
    redundant_cell = block_size - 1
    feature_tensor = np.zeros(
        [num_cell_y - redundant_cell, num_cell_x - redundant_cell, block_size * block_size * bins])
    for bx in range(num_cell_x - redundant_cell):  # 7
        for by in range(num_cell_y - redundant_cell):  # 15
            by_from = by
            by_to = by + block_size
            bx_from = bx
            bx_to = bx + block_size
            v = hist_tensor[by_from:by_to, bx_from:bx_to, :].flatten()  # to 1-D array (vector)
            feature_tensor[by, bx, :] = v / LA.norm(v, 2)
            # avoid NaN:
            if np.isnan(feature_tensor[by, bx, :]).any():  # avoid NaN (zero division)
                feature_tensor[by, bx, :] = v

    return feature_tensor.flatten()  # 3780 features


def read_image_with_pillow(img_path, is_gray=True):
    pil_im = Image.open(img_path).convert('RGB')
    img = np.array(pil_im)
    img = img[:, :, ::-1].copy()  # Convert RGB to BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def main(model_path, img_path):
    # load pretrained model
    # https://scikit-learn.org/stable/modules/model_persistence.html
    svm_model = joblib.load(model_path)

    time_start = time.time()

    # read image & extract HOG feature
    img = read_image_with_pillow(img_path, is_gray=True)
    img = cv2.resize(src=img, dsize=(64, 128))

    f = hog(img)

    # predict
    pred_y1 = svm_model.predict(np.array([f]))
    pred_y = svm_model.predict_proba(np.array([f]))

    class_probs = pred_y[0]
    max_class, max_prob = max(enumerate(class_probs), key=operator.itemgetter(1))

    class_str = 'PERSON' if max_class == 1 else 'BACKGROUND'
    prob_str = '%d' % int(max_prob * 100)

    time_end = time.time()

    print('------------------------------------------------------------------------')
    print('%s => Detected %s @ confidence: %s%% (elapsed time: %ss)' % (
    os.path.basename(img_path), class_str, prob_str, '%.2f' % (time_end - time_start)))
    print('------------------------------------------------------------------------')
    pass


if __name__ == "__main__":
    image_path = IMG_PATH if len(sys.argv) == 1 else sys.argv[1]
    main(MODEL_PATH, image_path)
    print('* Follow me @ ' + "\x1b[1;%dm" % (34) + ' https://www.facebook.com/minhng.info/' + "\x1b[0m")
    print('* Join GVGroup for discussion @ ' + "\x1b[1;%dm" % (
        34) + 'https://www.facebook.com/groups/ip.gvgroup/' + "\x1b[0m")
    print('* Thank you ^^~')
