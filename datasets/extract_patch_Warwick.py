import os
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
os.add_dll_directory("C:/Users/L_LE-BESCOND/AppData/Local/Programs/Python/openslide-win64-20171122/bin")
from utils import WholeSlideImage


def main():
    parser = argparse.ArgumentParser(description='Extract patches from Warwick dataset')
    parser.add_argument('data_path', type=str, default=None,
                        help='Data path')
    parser.add_argument('result_path', type=str, default=None,
                        help='Result path')
    parser.add_argument('--segment_contour', action="store_true",
                        help='Segment WSI tissue contours')
    parser.add_argument('--contour_snapchot', action="store_true",
                        help='Save contours snapchot along contour files')
    parser.add_argument('--snapchot_level', type=int, default=4,
                        help='Snapachot downsample level')
    parser.add_argument('--extract_patches', action="store_true",
                        help='Extract patches from tissue contours')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Patch size')
    parser.add_argument('--patch_level', type=int, default=0,
                        help='Patch downsample level')

    args = parser.parse_args()

    Path(args.result_path).mkdir(parents=True, exist_ok=True)

    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(args.data_path) for f in filenames if
             f.endswith("_HER2.ndpi") or f.endswith("Her2.ndpi")]
    for file in tqdm(files):
        WSI_object = WholeSlideImage(file)
        print(f"Processing slide {WSI_object.name} ...")
        if args.segment_contour:
            WSI_object.segmentTissue()
            print(f"... slide {WSI_object.name} segmented, now exporting ...")
            WSI_object.export_contours(args.result_path, args.contour_snapchot, args.snapchot_level)
        if args.extract_patches:
            df_contours = pd.read_csv(os.path.join(args.result_path, "contours.csv"))
            contours = WSI_object.load_element(os.path.join(args.result_path, WSI_object.name, "contours.pkl"))
            contours_idxs = df_contours[(df_contours.case_id == int(WSI_object.name)) &
                                        (df_contours.included == 1)].contours.values
            for contour_idx in contours_idxs:
                print(f"... Extracting patches from slide {WSI_object.name} from contour {contour_idx}")
                WSI_object.process_contour(contours[contour_idx], patch_level=args.patch_level, patch_size=args.patch_size,
                                           step_size=args.patch_size)
            print(f"... slide {WSI_object.name} patched, now exporting ... ")
            WSI_object.save_patches(args.result_path, patch_level=args.patch_level, patch_size=args.patch_size)
        print(f"Slide {WSI_object.name} done")


if __name__ == '__main__':
    main()
