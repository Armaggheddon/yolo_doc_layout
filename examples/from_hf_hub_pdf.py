from pathlib import Path
import pymupdf
from ultralytics import YOLO
import tqdm
from huggingface_hub import hf_hub_download

DOWNLOAD_PATH = Path(__file__).parent / "models"
PDF_PATH = Path(__file__).parent / "example.pdf"
RESULT_PATH = Path(__file__).parent / "results"
RESULT_PATH.mkdir(exist_ok=True)

available_models = [
    "yolo26n_doc_layout.pt",
    "yolo26s_doc_layout.pt",
    "yolo26m_doc_layout.pt",
]

model_path = hf_hub_download(
    repo_id="Armaggheddon/yolo26-document-layout",
    filename=available_models[0],  # Change index for different models
    repo_type="model",
    local_dir=DOWNLOAD_PATH,
)

# Initialize the model from the downloaded path
model = YOLO(model_path)


if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF file not found at {PDF_PATH}. Please provide a valid PDF file.")

# Convert PDF to images
pdf_document = pymupdf.open(PDF_PATH)
for page_num, page in tqdm.tqdm(
    enumerate(pdf_document), 
    total=len(pdf_document), 
    desc="Processing PDF pages"
):
    pix = page.get_pixmap(dpi=200) # if A4 should result in above 1400px width
    pil_img = pix.pil_image()
    result_img = model.predict(pil_img, imgsz=1280, save=False, verbose=False)
    result_img[0].save(filename=RESULT_PATH / f"page_{page_num + 1}.png")