from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2
import argparse

DOWNLOAD_PATH = Path(__file__).parent / "models"
SAMPLES_ROOT = Path(__file__).parent.parent / "plots/samples"

def run_inference(model_version, model_size):
    # Map version and size to HF repo and filename
    if model_version == "11":
        repo_id = "Armaggheddon/yolo11-document-layout"
        filename = f"yolo11{model_size}_doc_layout.pt"
    elif model_version == "26":
        # Placeholder for new YOLOv26 repo
        repo_id = "Armaggheddon/yolo26-document-layout"
        filename = f"yolo26{model_size}_doc_layout.pt"
    else:
        raise ValueError("Unsupported model version")

    print(f"Downloading {filename} from {repo_id}...")
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        local_dir=DOWNLOAD_PATH,
    )

    model = YOLO(model_path)
    images = list(SAMPLES_ROOT.glob("*.png"))
    images.sort()

    for i, img_path in enumerate(images):
        results = model(str(img_path), conf=0.5, verbose=False)
        annotated_frame = results[0].plot()
        output_path = SAMPLES_ROOT / f"{model_version}_{model_size}_annotated_{i+1}.png"
        cv2.imwrite(str(output_path), annotated_frame)
        print(f"Saved annotated image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using YOLO v11 or v26 from HF Hub")
    parser.add_argument("--version", type=str, choices=["11", "26"], default="26")
    parser.add_argument("--size", type=str, choices=["n", "s", "m"], default="n")
    args = parser.parse_args()
    
    run_inference(args.version, args.size)
