
import numpy as np
import copy
import cv2
import pywt
import os
import time
from PIL import Image

import torch.multiprocessing as mp


def merge_zstack(directory):
    #     print('{} Begin!!!'.format(directory))
    # Get a list of the image files
    image_files = sorted([f for f in os.listdir(directory) if (f.endswith('.png') or f.endswith('.jpg')) and f.startswith('20')])

    # Read the images
    images = [cv2.imread(os.path.join(directory, img)) for img in image_files]

    # Initialize final_coeffs for each channel
    first_wt = pywt.wavedec2(cv2.split(images[0])[0], 'db1', level=2)

    # Create deep copies of the initial coefficients for each channel
    final_coeffs_R = copy.deepcopy(first_wt)
    final_coeffs_G = copy.deepcopy(first_wt)
    final_coeffs_B = copy.deepcopy(first_wt)

    # Loop over each channel
    for channel in range(3):
        wavelet_transforms = [pywt.wavedec2(cv2.split(img)[channel], 'db1', level=2) for img in images]
        for i in range(len(wavelet_transforms[0])):
            for j in range(len(wavelet_transforms[0][i])):
                max_coeff = np.max([wt[i][j] for wt in wavelet_transforms], axis=0)
                if channel == 0:
                    final_coeffs_R[i][j][:] = max_coeff
                elif channel == 1:
                    final_coeffs_G[i][j][:] = max_coeff
                else:
                    final_coeffs_B[i][j][:] = max_coeff

    # Compute inverse wavelet transform for each channel
    final_image_R = pywt.waverec2(final_coeffs_R, 'db1')
    final_image_G = pywt.waverec2(final_coeffs_G, 'db1')
    final_image_B = pywt.waverec2(final_coeffs_B, 'db1')

    # Make sure they are in uint8 format and in the range [0, 255]
    final_image_R = np.clip(final_image_R, 0, 255).astype('uint8')
    final_image_G = np.clip(final_image_G, 0, 255).astype('uint8')
    final_image_B = np.clip(final_image_B, 0, 255).astype('uint8')

    # Merge channels back but in RGB order
    final_image = cv2.merge((final_image_B, final_image_G, final_image_R))  # Changed the order here
    final_image = Image.fromarray(final_image, 'RGB')

    # # save image to the corresponding folder
    # im = final_image
    # im = im.resize((128, 128))
    #
    # dirtmp = 'E:/data/Porcine_Exp_Davis_Processed_All/exp_23/' + directory.split('/')[-2] + '/'
    # if not os.path.exists(dirtmp):
    #     os.makedirs(dirtmp)
    # im.save(dirtmp + '{}.png'.format(directory.split('/')[-1]))
    # print("Saved : {} + /../../ + merged_{}.jpg".format(directory, directory.split('/')[-1]))

    return final_image, final_coeffs_R, final_coeffs_G, final_coeffs_B


def merge_zstack1(directory):
    #     print('{} Begin!!!'.format(directory))
    # Get a list of the image files
    image_files = sorted([f for f in os.listdir(directory) if (f.endswith('.png') or f.endswith('.jpg')) and f.startswith('20')])

    # Read the images
    images = [cv2.imread(os.path.join(directory, img)) for img in image_files]

    # Initialize final_coeffs for each channel
    first_wt = pywt.wavedec2(cv2.split(images[0])[0], 'db1', level=2)

    # Create deep copies of the initial coefficients for each channel
    final_coeffs_R = copy.deepcopy(first_wt)
    final_coeffs_G = copy.deepcopy(first_wt)
    final_coeffs_B = copy.deepcopy(first_wt)

    # Loop over each channel
    for channel in range(3):
        wavelet_transforms = [pywt.wavedec2(cv2.split(img)[channel], 'db1', level=2) for img in images]
        for i in range(len(wavelet_transforms[0])):
            for j in range(len(wavelet_transforms[0][i])):
                max_coeff = np.max([wt[i][j] for wt in wavelet_transforms], axis=0)
                if channel == 0:
                    final_coeffs_R[i][j][:] = max_coeff
                elif channel == 1:
                    final_coeffs_G[i][j][:] = max_coeff
                else:
                    final_coeffs_B[i][j][:] = max_coeff

    # Compute inverse wavelet transform for each channel
    final_image_R = pywt.waverec2(final_coeffs_R, 'db1')
    final_image_G = pywt.waverec2(final_coeffs_G, 'db1')
    final_image_B = pywt.waverec2(final_coeffs_B, 'db1')

    # Make sure they are in uint8 format and in the range [0, 255]
    final_image_R = np.clip(final_image_R, 0, 255).astype('uint8')
    final_image_G = np.clip(final_image_G, 0, 255).astype('uint8')
    final_image_B = np.clip(final_image_B, 0, 255).astype('uint8')

    # Merge channels back but in RGB order
    final_image = cv2.merge((final_image_B, final_image_G, final_image_R))  # Changed the order here
    final_image = Image.fromarray(final_image, 'RGB')

    # save image to the corresponding folder
    im = final_image
    im = im.resize((128, 128))

    dirtmp = 'E:/data/Porcine_Exp_Davis_Processed_All/exp_23/' + directory.split('/')[-2] + '/'
    if not os.path.exists(dirtmp):
        os.makedirs(dirtmp)
    im.save(dirtmp + '{}.png'.format(directory.split('/')[-1]))
    print("Saved : {} + /../../ + merged_{}.jpg".format(directory, directory.split('/')[-1]))

    return final_image, final_coeffs_R, final_coeffs_G, final_coeffs_B


def imageA(imDir, im):
    img = cv2.imread(imDir)
    assert img is not None, "file could not be read, check with os.path.exists()"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [0, 255, 0]

    markers = markers.astype(np.uint8)

    ret, m2 = cv2.threshold(markers, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(m2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M['m00'] != 0.0:
            x1 = int(M['m10'] / M['m00'])
            y1 = int(M['m01'] / M['m00'])

        area = cv2.contourArea(cnt)
        # Convert the area from pixels to a real-world unit of measurement (e.g. cm^2)
        scale_factor = 0.0025  # 1 pixel = 0.0025 cm
        size = area * scale_factor ** 2
        size = round(size, 2)
        if size > 4 or size <= 0.01:
            continue

        perimeter = cv2.arcLength(cnt, True)
        perimeter = perimeter * scale_factor
        perimeter = round(perimeter, 2)
        print(f'Area of contour {i + 1}:', size)
        print(f'Perimeter of contour {i + 1}:', perimeter)
        img = cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)
        cv2.putText(img, f'Area: {size} cm2', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f'Perimeter: {perimeter} cm', (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the image with the contours drawn
    cv2.imwrite(imDir[:-4] + '_Object.jpg', img)
    cv2.waitKey(0)

    return area

if __name__ == "__main__":

    desktop_dir_main = 'E:/Data/Porcine_Exp_Davis/exp_23/'
    for ddir in os.listdir(desktop_dir_main):
        desktop_dir = desktop_dir_main + ddir + '/'

        all_dirs = []
        for tmp in os.listdir(desktop_dir):
            # daytmp = "Day {}".format(i)
            # im_tmp = desktop_dir + daytmp
            # for ca in os.listdir(im_tmp):
            #     im_tmp += ca
            #     if os.path.isdir(im_tmp):
            #         all_dirs.append(im_tmp)
            all_dirs.append(desktop_dir + tmp)
        nprc = 5
        print(desktop_dir)
        for im in range(int(len(all_dirs) / nprc) + 2):
            process = []
            for i in range(nprc):
                time.sleep(0.1)
                if nprc * im + i < len(all_dirs):
                    p = mp.Process(target=merge_zstack1, args=(all_dirs[nprc * im + i],))
                    p.start()
                    process.append(p)

            for p in process:
                p.join()
                time.sleep(0.1)