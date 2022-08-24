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

from utils.utils import readImageAndGetClicks, getClickMapAndBoundingBox, \
    getPatchs, sharpnessEnhancement, contrastEnhancement, \
    predictPatchs, postProcessing, generateInstanceMap, readImageAndGetSignals, predictSingleImage

seeddd = 1
img_rows = config.img_rows  # 480#640
img_cols = config.img_cols  # 768#1024
img_chnls = config.img_chnls
input_shape = (img_rows, img_cols)
testTimeAug = config.testTimeAug


def custom_get_click(img_path, mask_path):
    img = Image.open(mask_path)
    hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
    yellow_lower = np.array([20, 190, 150])
    yellow_upper = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    label_image = label(mask)
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

    data_dir = "..."
    imgPaths = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if not file.startswith("Inked")]
    maskPaths = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.startswith("Inked")]
    imgPaths.sort()
    maskPaths.sort()
    for imgPath, maskPath in zip(imgPaths, maskPaths):
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
            imsave(imgPath[:-4] + '_overlay.png', instanceMap_RGB)
            imsave(imgPath[:-4] + '_instances.png', instanceMap * 255)
            imsave(imgPath[:-4] + '_points.png', np.uint8(255 * np.sum(nucPoints, axis=(0, 3))))
            # plt.figure(),plt.imshow(img)

        if config.application == 'Gland':
            img, markups, imgPath = readImageAndGetSignals(os.getcwd())
            instanceMap = predictSingleImage(model, img, markups)
            instanceMap_RGB = label2rgb(np.uint8(instanceMap), image=img, alpha=0.3, bg_label=0, bg_color=(0, 0, 0),
                                        image_alpha=1, kind='overlay')
            plt.figure(), plt.imshow(instanceMap_RGB), plt.show()
            imsave(imgPath[:-4] + '_instances.png', instanceMap)
            imsave(imgPath[:-4] + '_signals.png', markups)


if __name__ == '__main__':
    main()