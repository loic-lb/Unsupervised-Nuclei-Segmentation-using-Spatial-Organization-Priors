import os
import cv2
import re

import openslide
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from PIL import Image

from .wsiutils import local_average, compute_law_feats, filter_ROI, thresh_ROI, \
    floodfill_ROI, contour_ROI, remove_black_areas, isWhitePatch_HLS, isInContourV3_Easy, isInContourV3_Hard


class WholeSlideImage(object):
    def __init__(self, path):

        """
        Args:
            path (str): fullpath to WSI file
            ROIsdf (DataFrame): df with ROI coordinates for each slide
        """
        self.path = path
        self.name = str(re.search(r'(?:[^\\\/](?!(\\|\/)))+(?=.ndpi)', path).group(0).split("_")[0])
        self.wsi = openslide.open_slide(path)
        self.level_downsamples = self.wsi.level_downsamples
        self.level_dim = self.wsi.level_dimensions

        self.contours_tissue = None
        self.patches = []
        self.count_patch_in_contour = 0

    def getOpenSlide(self):
        return self.wsi

    def load_slide(self, seg_level):
        slide = self.wsi.read_region((0, 0), seg_level, self.level_dim[seg_level]).convert("RGB")
        slide = remove_black_areas(slide)
        return slide

    def extract_region(self, seg_level, contour, color_contour=True):
        slide = self.load_slide(seg_level)
        coeff = self.level_downsamples[seg_level]
        slide = np.asarray(slide)
        if color_contour:
            cv2.drawContours(slide, [(contour // coeff).astype(np.int32)], -1, (125, 0, 0), 3)
        l, t, r, b = (
        np.min(contour[:, :, 0]), np.min(contour[:, :, 1]), np.max(contour[:, :, 0]), np.max(contour[:, :, 1]))
        return Image.fromarray(slide).crop((l // coeff, t // coeff, r // coeff, b // coeff))

    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def save_element(filepath, element):
        open_file = open(filepath, "wb")
        pickle.dump(element, open_file)
        open_file.close()
        return

    @staticmethod
    def load_element(filepath):
        open_file = open(filepath, "rb")
        element = pickle.load(open_file)
        open_file.close()
        return element

    def segmentTissue(self, seg_level=5, window_avg=3, window_eng=3, thresh=90, keep_grayscale=False, inv=False,
                      start=(50, 50), area_min=5e3):
        contours_slide = []

        img = self.load_slide(seg_level)
        img_avg = local_average(np.asarray(img), window_avg, keep_grayscale)
        law_feats = compute_law_feats(img_avg, window_eng)
        filterred_roi = filter_ROI(law_feats[:, :, 3].astype(np.uint8()))
        threshed_roi = thresh_ROI(filterred_roi, thresh, inv)
        flooded_roi = floodfill_ROI(threshed_roi, start)
        contours = contour_ROI(flooded_roi)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > area_min:
                contours_slide.append(contour)
        if len(contours_slide) == 0:
            self.contours_tissue = []
            print(f"No contours found for slide {self.name}")
            return
        scale = self.level_downsamples[seg_level]
        self.contours_tissue = self.scaleContourDim(contours_slide, scale)
        return

    def export_contours(self, output_dir, snapchot=True, snapchot_level=4):
        if os.path.isfile(os.path.join(output_dir, "contours.csv")):
            df = pd.read_csv(os.path.join(output_dir, "contours.csv"))
        else:
            df = pd.DataFrame(columns=['case_id', 'contours', 'included'])
        work_path = os.path.join(output_dir, self.name)
        Path(work_path).mkdir(parents=True, exist_ok=True)
        for i, contour in enumerate(self.contours_tissue):
            df = pd.concat([df, pd.DataFrame({'case_id': [self.name], 'contours': [i], 'included': [1]},
                                             columns=df.columns)], ignore_index=True)
            if snapchot:
                slide_region = self.extract_region(snapchot_level, contour)
                slide_region.save(os.path.join(work_path, f"contour_{i}.png"))
        self.save_element(os.path.join(work_path, "contours.pkl"), self.contours_tissue)
        df.to_csv(os.path.join(output_dir, "contours.csv"), index=False)

    @staticmethod
    def is_white(patch):
        return isWhitePatch_HLS(patch, lightThresh=220, percentage=0.8)

    def process_coord_candidate(self, coord, cont_check_fn, path, patch_level, patch_size, pos):
        if cont_check_fn(coord):
            self.count_patch_in_contour += 1
            wsi = openslide.open_slide(path)
            patch = wsi.read_region(coord, patch_level, tuple([patch_size, patch_size])).convert('RGB')
            if not self.is_white(patch):
                return (coord, pos)
            else:
                return None
        else:
            return None

    def process_contour(self, cont, patch_level, patch_size=256, step_size=256,
                        contour_fn='easy', use_padding=True, top_left=None, bot_right=None):
        self.count_patch_in_contour = 0
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (
            0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])
        patch_downsample = (int(self.level_downsamples[patch_level]), int(self.level_downsamples[patch_level]))
        ref_patch_size = (patch_size * patch_downsample[0], patch_size * patch_downsample[0])

        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y + h
            stop_x = start_x + w
        else:
            stop_y = min(start_y + h, img_h - ref_patch_size[1] + 1)
            stop_x = min(start_x + w, img_w - ref_patch_size[0] + 1)

        if bot_right is not None:
            stop_y = min(bot_right[1], stop_y)
            stop_x = min(bot_right[0], stop_x)
        if top_left is not None:
            start_y = max(top_left[1], start_y)
            start_x = max(top_left[0], start_x)

        if bot_right is not None or top_left is not None:
            w, h = stop_x - start_x, stop_y - start_y
            if w <= 0 or h <= 0:
                print("Contour is not in specified ROI, skip")
                return {}, {}

        if contour_fn == "easy":
            cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
        elif contour_fn == "hard":
            cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
        else:
            raise NotImplementedError

        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]

        x_range = np.arange(start_x, stop_x, step=step_size_x)
        y_range = np.arange(start_y, stop_y, step=step_size_y)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()
        x_pos, y_pos = np.meshgrid(np.arange(len(x_range)), np.arange(len(y_range)), indexing='ij')
        pos_candidates = np.array([x_pos.flatten(), y_pos.flatten()]).transpose()

        num_workers = mp.cpu_count()
        pool = mp.Pool(num_workers)

        iterable = [(coord, cont_check_fn, self.path, patch_level, patch_size, pos_candidates[i]) for
                    i, coord in enumerate(coord_candidates)]

        results = pool.starmap(self.process_coord_candidate, iterable)
        pool.close()
        results = np.array([result for result in results if result is not None])

        print(
            f"Removed {np.round((1 - len(results) / self.count_patch_in_contour) * 100, 2)}% patches after filtering white patches")
        self.patches.append(results)

    def save_patches(self, output_dir, patch_level, patch_size):
        for patches_contour in self.patches:
            print(len(patches_contour))
        work_path = os.path.join(output_dir, self.name, "Patches")
        Path(work_path).mkdir(parents=True, exist_ok=True)
        for i, patches_contour in enumerate(self.patches):
            for c, pos in patches_contour:
                patch = self.wsi.read_region(c, patch_level, tuple([patch_size, patch_size])).convert('RGB')
                patch.save(os.path.join(work_path, f"roi-{pos[0]}_{pos[1]}.png"))
