import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Read the image and change it into a gray image
def preprocess(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    img = clahe.apply(img)
    return img

img_target = preprocess('Data/Test/target 5.jpg')

# Import the database
logo_path = 'Data/Logo'
samples = []
img_desc = []
img = os.listdir(logo_path)
for i in img:    
    clahe_img = preprocess(logo_path + '/' + i)
    samples.append(clahe_img)
    if "allianz" in i:
        img_desc.append(cv2.imread('Data/Desc/Allianz.jpg'))
    elif "unilever" in i:
        img_desc.append(cv2.imread('Data/Desc/Unilefer.jpg'))
    elif "mayora" in i:
        img_desc.append(cv2.imread('Data/Desc/Mayora.jpg'))
    elif "bni" in i:
        img_desc.append(cv2.imread('Data/Desc/BNI.jpg'))
    elif "krakatau" in i:
        img_desc.append(cv2.imread('Data/Desc/Krakatau.jpg'))
    elif "bri" in i:
        img_desc.append(cv2.imread('Data/Desc/BRI.jpg'))
    

def createMask(mask, match):
    total = 0
    for i, (f, s) in enumerate(match):
        if f.distance < 0.7 * s.distance:
            mask[i] = [1,0]
            total += 1
    return total, mask

def computeSIFT(samples):
    SIFT = cv2.SIFT_create()
    sift_target_kp, sift_target_ds = SIFT.detectAndCompute(img_target,None)
    sift_target_ds = np.float32(sift_target_ds)

    match_res = []

    max = 0
    number = 0
    i = 0

    for sample in samples:
        flann = cv2.FlannBasedMatcher(dict(algorithm = 1), dict(checks = 50))

        sift_sample_kp, sift_sample_ds = SIFT.detectAndCompute(sample,None)
        sift_sample_ds = np.float32(sift_sample_ds)
        sift_match = flann.knnMatch(sift_target_ds, sift_sample_ds, 2)
        sift_match_mask = [[0,0] for i in range (0,len(sift_match))]
        total, sift_match_mask = createMask(sift_match_mask, sift_match)

        sift_res = cv2.drawMatchesKnn(
            img_target,
            sift_target_kp,
            sample,
            sift_sample_kp,
            sift_match, None,
            matchColor=[255,0,0],
            singlePointColor=[0,255,0], matchesMask=sift_match_mask
        )
        match_res.append(sift_res)

        if total > max:
            number = i
            max = total
        i += 1

    _, axs = plt.subplots(2)
    axs[0].imshow(match_res[number])
    axs[0].set_title('Match Result')
    axs[0].axis('off')

    axs[1].imshow(img_desc[number])
    axs[1].axis('off')
    axs[1].set_title('Description')
    
    plt.show()

computeSIFT(samples)