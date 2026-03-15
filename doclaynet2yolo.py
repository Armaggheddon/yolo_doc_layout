from pathlib import Path
import json
from collections import defaultdict
import tqdm
import shutil

DATASET_ROOT = Path(__file__).parent / "DocLayNet_v1.2"
IMAGES_PATH = DATASET_ROOT / "PNG"
TEST_ANN_PATH = DATASET_ROOT / "COCO" / "test.json"
TRAIN_ANN_PATH = DATASET_ROOT / "COCO" / "train.json"
VAL_ANN_PATH = DATASET_ROOT / "COCO" / "val.json"


ROOT = Path(__file__).parent
OUT_DATASET_ROOT = ROOT / "dataset"
OUT_TRAIN = OUT_DATASET_ROOT / "train"
OUT_TEST = OUT_DATASET_ROOT / "test"
OUT_VAL = OUT_DATASET_ROOT / "val"

OUT_DATASET_ROOT.mkdir(exist_ok=True)
OUT_TRAIN.mkdir(exist_ok=True)
OUT_TEST.mkdir(exist_ok=True)
OUT_VAL.mkdir(exist_ok=True)

def parse_split(
    images_path: Path,
    ann_path: Path,
    dst_path: Path,
    copy: bool = False
) -> None:
    # If copy = True copies the images from ann-path to dst_path/"images"
    # else moves them and deletes the original.
    
    data = json.load(open(ann_path, "r"))
    
    id2category = {
        ann["id"]-1: ann["name"] # -1 to have 0-indexed
        for ann in data["categories"]
    }
    
    annotations = defaultdict(list) # map with image_id -> data[annotations][x]
    for ann in data["annotations"]:
        if ann["precedence"] != 0:
            # it means that is a duplicate annotation and has less
            # relevence wrt the one with precedence=0
            continue
        annotations[ann["image_id"]].append(ann)
    
    for img in tqdm.tqdm(data["images"]):
        if img["precedence"] != 0:
            continue
        img_id = img["id"]
        img_w, img_h = img["width"], img["height"]
        filename = img["file_name"]
        
        complete_img_path = images_path / filename
        if not complete_img_path.exists():
            print(f"Image file {filename} not found in {images_path}")
            continue
        
        file_lines = []
        
        img_anns = annotations[img_id]
        # build the .txt with the annotations
        for ann in img_anns:
            category_id = ann["category_id"]
            bbox = ann["bbox"] # (x, y, w, h)
            # convert to xc, yx, w, h
            bbox[0] = bbox[0] + bbox[2] / 2
            bbox[1] = bbox[1] + bbox[3] / 2
            
            bbox[0] /= img_w
            bbox[1] /= img_h
            bbox[2] /= img_w
            bbox[3] /= img_h # normalize 0-1
            file_lines.append(
                f"{category_id-1} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"
            )
        
        if copy:
            shutil.copy(
                complete_img_path,
                dst_path / filename
            )
        else:
            shutil.move(
                complete_img_path,
                dst_path / filename
            )
            
        ann_filename = filename.rsplit(".")[0] + ".txt"
        ann_file_path = dst_path / ann_filename
        
        with open(ann_file_path, "w") as f:
            f.write("\n".join(file_lines))
    

if __name__ == "__main__":
    
    print(f"Converting Train split: ")
    parse_split(
        IMAGES_PATH,
        TRAIN_ANN_PATH,
        OUT_TRAIN
    )
    
    print(f"Converting Test split: ")
    parse_split(
        IMAGES_PATH,
        TEST_ANN_PATH,
        OUT_TEST
    )
    
    print(f"Converting Validation split: ")
    parse_split(
        IMAGES_PATH,
        VAL_ANN_PATH,
        OUT_VAL
    )
            
            
        
        
    
    