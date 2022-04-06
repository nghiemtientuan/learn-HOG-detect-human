import os
import random
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
"""

TRAIN_POS_LST = 'INRIAPerson/train_64x128_H96/pos.lst'
TRAIN_POS_DIR = 'INRIAPerson/96X160H96/Train'

TRAIN_NEG_NUM_PATCHES_PER_IMAGE = 10
TRAIN_NEG_LST = 'INRIAPerson/Train/neg.lst'
TRAIN_NEG_DIR = 'INRIAPerson/Train'

TRAIN_NEG_PATCH_SIZE_RANGE = (0.4, 1.0)


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


def train(train_pos_lst, train_pos_dir, train_neg_lst, train_neg_dir, train_neg_num_patches_per_image,
          train_neg_patch_size_range):
    assert os.path.isfile(train_pos_lst) and os.path.isfile(train_neg_lst)

    # ---------- READ & EXTRACT POSITIVE SAMPLES (PERSON) ----------
    with open(train_pos_lst) as f:
        pos_lines = f.readlines()

    positive_features = []
    pos_lines = [os.path.join(train_pos_dir, '/'.join(pl.split('/')[1:])).strip() for pl in pos_lines]
    for idx, pline in enumerate(pos_lines):
        img_path = pline
        if not os.path.isfile(img_path):
            print('[pos] Skipped %s' % img_path)
            continue
        img = read_image_with_pillow(img_path, is_gray=True)
        img = cv2.resize(src=img, dsize=(64, 128))
        f = hog(img)
        positive_features.append(f)
        print('[pos][%d/%d] Done HOG feature extraction @ %s' % (idx + 1, len(pos_lines), img_path))

    positive_features = np.array(positive_features)
    # ---------- END - READ & EXTRACT POSITIVE SAMPLES (PERSON) ----------

    # ---------- READ & EXTRACT NEGATIVE SAMPLES (BACKGROUND) ----------
    with open(train_neg_lst) as f:
        neg_lines = f.readlines()

    negative_features = []
    neg_lines = [os.path.join(train_neg_dir, '/'.join(pl.split('/')[1:])).strip() for pl in neg_lines]
    for idx, nline in enumerate(neg_lines):
        img_path = nline
        if not os.path.isfile(img_path):
            print('[neg] Skipped %s' % img_path)
            continue
        img = read_image_with_pillow(img_path, is_gray=True)
        img_h, img_w = img.shape
        img_min_size = min(img_h, img_w)

        # random crop
        negative_patches = []
        for num_neg_idx in range(train_neg_num_patches_per_image):
            random_patch_size = random.uniform(train_neg_patch_size_range[0], train_neg_patch_size_range[1])
            random_patch_height = int(random_patch_size * img_min_size)
            random_patch_width = int(random_patch_height * random.uniform(0.3, 0.7))
            random_position_x = random.randint(0, img_w - random_patch_width)
            random_position_y = random.randint(0, img_h - random_patch_height)
            # crop image -> image patch
            npatch = img[random_position_y:random_position_y + random_patch_height,
                     random_position_x:random_position_x + random_patch_width]
            #             cv2.imwrite('npatch-%d.jpg' % num_neg_idx, npatch)
            negative_patches.append(npatch)

        for npatch in negative_patches:
            img = cv2.resize(src=npatch, dsize=(64, 128))
            f = hog(img)
            negative_features.append(f)
        print('[neg][%d/%d] Done HOG feature extraction @ %s' % (idx + 1, len(pos_lines), img_path))

    negative_features = np.array(negative_features)
    # ---------- END - READ & EXTRACT NEGATIVE SAMPLES (BACKGROUND) ----------

    print('Our positive features matrix: ', positive_features.shape)  # (2416, 3780)
    print('Our negative features matrix: ', negative_features.shape)  # (12180, 3780)

    x = np.concatenate((negative_features, positive_features), axis=0)  # (14596, 3730)
    y = np.array([0] * negative_features.shape[0] + [1] * positive_features.shape[0])

    print('X: ', x.shape)  # (14596, 3780)
    print('Y: ', y.shape)  # (14596,)
    print('Start training model with X & Y samples...')

    # ---------- TRAIN SVM ----------

    # https://scikit-learn.org/stable/modules/svm.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    model = svm.SVC(C=0.01, kernel='rbf', probability=True)
    model = model.fit(x, y)

    print('Done training model!')
    return model


def main():
    svm_model = train(train_pos_lst=TRAIN_POS_LST,
                      train_pos_dir=TRAIN_POS_DIR,
                      train_neg_lst=TRAIN_NEG_LST,
                      train_neg_dir=TRAIN_NEG_DIR,
                      train_neg_num_patches_per_image=TRAIN_NEG_NUM_PATCHES_PER_IMAGE,
                      train_neg_patch_size_range=TRAIN_NEG_PATCH_SIZE_RANGE)

    # save model
    # https://scikit-learn.org/stable/modules/model_persistence.html
    out_model_name = 'model_hog_person.joblib'
    joblib.dump(svm_model, out_model_name)
    print('=> Trained model is saved @ %s' % out_model_name)
    pass


if __name__ == "__main__":
    #     print('Start running HOG on image @ %s' % IMG)
    main()
    print('* Follow me @ ' + "\x1b[1;%dm" % (34) + ' https://www.facebook.com/minhng.info/' + "\x1b[0m")
    print('* Join GVGroup for discussion @ ' + "\x1b[1;%dm" % (
        34) + 'https://www.facebook.com/groups/ip.gvgroup/' + "\x1b[0m")
    print('* Thank you ^^~')
