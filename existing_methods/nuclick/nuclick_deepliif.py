import numpy as np
from skimage.io import imsave
import matplotlib.pyplot as plt
from models.models import getModel
from skimage.color import label2rgb
from config import config
import os
from PIL import Image
import cv2
from skimage.measure import label, regionprops
import time

from utils.utils import readImageAndGetClicks, getClickMapAndBoundingBox, \
    getPatchs, sharpnessEnhancement, contrastEnhancement, \
    predictPatchs, postProcessing, generateInstanceMap, readImageAndGetSignals, predictSingleImage

seeddd = 1
img_rows = config.img_rows  # 480#640
img_cols = config.img_cols  # 768#1024
img_chnls = config.img_chnls
input_shape = (img_rows, img_cols)
testTimeAug = config.testTimeAug

def process_mask(img):
    img = np.array(img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180) for red
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    lower_range = np.array([110, 50, 50])
    upper_range = np.array([130, 255, 255])
    mask2 = cv2.inRange(img_hsv, lower_range, upper_range)

    # join my masks
    mask = mask0 + mask1 + mask2

    return ((mask > 0) * 255.).astype(np.uint8)


def custom_get_click(img_path, mask_path):
    mask = Image.open(mask_path)
    label_image = label(process_mask(mask))
    pts = np.array([region.centroid for region in regionprops(label_image)])
    img_return = cv2.imread(img_path)
    return img_return[:, :, ::-1], np.round(pts[:, 1]).astype(int), np.round(pts[:, 0]).astype(int)


def main():
    modelType = config.modelType  # ['MultiScaleResUnet']
    lossType = config.lossType
    modelBaseName = 'NuClick_%s_%s_%s' % (config.application, modelType, lossType)
    modelSaveName = "%s/weights-%s.h5" % (config.weights_path, modelBaseName)

    # loading models
    model = getModel(modelType, lossType, input_shape)
    model.load_weights(modelSaveName)

    data_dir = ".../DeepLIIF_Testing/"
    result_dir = ".../results/"
    mask_dir = os.path.join(data_dir, "ground_truths")
    imgPaths = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".png")]
    maskPaths = [os.path.join(mask_dir, file) for file in os.listdir(mask_dir) if file.endswith(".png")]
    imgPaths.sort()
    maskPaths.sort()
    avg_time = 0
    for imgPath, maskPath in zip(imgPaths, maskPaths):
        start_time = time.time()
        print(imgPath,maskPath)
        if config.application in ['Cell', 'Nucleus']:
            img, cx, cy = custom_get_click(imgPath, maskPath)
            m, n = img.shape[0:2]
            clickMap, boundingBoxes = getClickMapAndBoundingBox(cx, cy, m, n)
            patchs, nucPoints, otherPoints = getPatchs(img, clickMap, boundingBoxes, cx, cy, m, n)
            dists = np.float32(
                np.concatenate((nucPoints, otherPoints, otherPoints), axis=3))  # the last one is only dummy!
            # prediction with test time augmentation
            predNum = 0  # augNum*numModel
            preds = np.zeros((len(patchs), img_rows, img_cols), dtype=np.float32)
            preds += predictPatchs(model, patchs, dists, config.testTimeJittering)
            predNum += 1
            print("Original images prediction, DONE!")
            if testTimeAug:
                print("Test Time Augmentation Started")
                # sharpenning the image
                patchs_shappened = patchs.copy()
                for i in range(len(patchs)):
                    patchs_shappened[i] = sharpnessEnhancement(patchs[i])
                temp = predictPatchs(model, patchs_shappened[:, :, ::-1], dists[:, :, ::-1], config.testTimeJittering)
                preds += temp[:, :, ::-1]
                predNum += 1
                print("Sharpenned images prediction, DONE!")

                # contrast enhancing the image
                patchs_contrasted = patchs.copy()
                for i in range(len(patchs)):
                    patchs_contrasted[i] = contrastEnhancement(patchs[i])
                temp = predictPatchs(model, patchs_contrasted[:, ::-1, ::-1], dists[:, ::-1, ::-1],
                                     config.testTimeJittering)
                preds += temp[:, ::-1, ::-1]
                predNum += 1
                print("Contrasted images prediction, DONE!")
            preds /= predNum
            try:
                masks = postProcessing(preds, thresh=config.Thresh, minSize=config.minSize, minHole=config.minHole,
                                       doReconstruction=True, nucPoints=nucPoints)
            except:
                masks = postProcessing(preds, thresh=config.Thresh, minSize=config.minSize, minHole=config.minHole,
                                       doReconstruction=False, nucPoints=nucPoints)
            instanceMap = generateInstanceMap(masks, boundingBoxes, m, n)
            instanceMap_RGB = label2rgb(instanceMap, image=img, alpha=0.3, bg_label=0, bg_color=(0, 0, 0),
                                        image_alpha=1,
                                        kind='overlay')
            # plt.figure(), plt.imshow(instanceMap_RGB)
            # plt.show()
            imsave(os.path.join(result_dir,imgPath.split("/")[-1].split(".")[0]) + '_overlay.png', instanceMap_RGB)
            imsave(os.path.join(result_dir,imgPath.split("/")[-1].split(".")[0]) + '_instances.png', instanceMap * 255)
            imsave(os.path.join(result_dir,imgPath.split("/")[-1].split(".")[0]) + '_points.png', np.uint8(255 * np.sum(nucPoints, axis=(0, 3))))
            # plt.figure(),plt.imshow(img)
        end_time = time.time()
        avg_time += (end_time-start_time)
        print("Time for inference = "+str(end_time-start_time)+" seconds")
        if config.application == 'Gland':
            img, markups, imgPath = readImageAndGetSignals(os.getcwd())
            instanceMap = predictSingleImage(model, img, markups)
            instanceMap_RGB = label2rgb(np.uint8(instanceMap), image=img, alpha=0.3, bg_label=0, bg_color=(0, 0, 0),
                                        image_alpha=1, kind='overlay')
            plt.figure(), plt.imshow(instanceMap_RGB), plt.show()
            imsave(imgPath[:-4] + '_instances.png', instanceMap)
            imsave(imgPath[:-4] + '_signals.png', markups)
    print("Average time by inference = "+str(avg_time/len(imgPaths))+" seconds")
if __name__ == '__main__':
    main()