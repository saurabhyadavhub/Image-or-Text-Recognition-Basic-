# AI Internship Final Project — Image Text Recognition (OCR)

**Project 4 | DecodeLabs-style brief: Image or Text Recognition (Basic)**

A runnable Python computer-vision project that extracts text from an image. It uses OpenCV to preprocess the image, Tesseract OCR through `pytesseract` to recognize words, and a confidence filter to keep tokens with confidence **at least 80%** by default.

## Features
- Reads PNG/JPG image inputs.
- Converts the image to grayscale, reduces noise with Gaussian blur, and applies Otsu thresholding.
- Uses Tesseract OCR (`--oem 3 --psm 6`).
- Filters recognized words using a configurable confidence threshold (default 80%).
- Saves the preprocessed image, annotated image, extracted text, and a JSON report.

## Project structure
```text
AI_Internship_Final_Project/
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── samples/
│   └── sample_text.png
└── outputs/
    └── .gitkeep
```

## 1. Requirements
- Python 3.10 or newer
- Tesseract OCR engine installed separately
- Python packages from `requirements.txt`

### Install Tesseract
- **Windows:** Install the Tesseract OCR Windows package, then add its installation folder to PATH. If needed, pass its full path using `--tesseract-cmd`.
- **Ubuntu/Debian:** `sudo apt update && sudo apt install tesseract-ocr`
- **macOS (Homebrew):** `brew install tesseract`

## 2. Setup
Open a terminal in this project folder:

```bash
python -m venv .venv
```

Activate the environment:
- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

## 3. Run the project
Run the included sample:
```bash
python main.py
```

Use your own image:
```bash
python main.py --image path/to/your_image.jpg
```

Change the confidence threshold:
```bash
python main.py --image samples/sample_text.png --threshold 70
```

If Tesseract is installed but not on PATH (Windows example):
```bash
python main.py --image samples/sample_text.png --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## 4. Output files
After a successful run, files are saved in `outputs/`:
- `*_preprocessed.png` — black-and-white image supplied to OCR.
- `*_annotated.jpg` — original image with boxes and confidence labels for accepted words.
- `*_recognized.txt` — text tokens that passed the confidence threshold.
- `*_results.json` — accepted/rejected tokens, bounding boxes, confidence values, and summary.

The confidence score is Tesseract's OCR estimate, not a guarantee of correctness. The default 80% threshold follows the project brief's confidence-filter milestone. If no tokens pass, try a clearer image or a lower threshold.

## 5. How it works
1. **Input:** load the selected image.
2. **Preprocessing:** grayscale → Gaussian blur → Otsu threshold.
3. **Recognition:** Tesseract returns text, bounding boxes, and confidence scores.
4. **Filtering:** keep tokens whose confidence is greater than or equal to the selected threshold.
5. **Output:** save recognized text and visual/report artifacts.

## Troubleshooting
- `TesseractNotFoundError`: install the Tesseract engine and check PATH.
- No text detected: use a sharper, well-lit image with readable text; try `--threshold 60`.
- Incorrect layout: change the page segmentation mode in `main.py` (`--psm 6` is suitable for a block of text).
- OpenCV cannot read the file: confirm the image path and use PNG/JPG.

## Limitations
OCR accuracy depends on image quality, font, language, orientation, and Tesseract installation. This basic project recognizes printed text; it is not a custom-trained model and does not guarantee perfect extraction.

## GitHub upload
1. Create a new repository on GitHub.
2. Extract this ZIP file.
3. Upload the project files (or run `git init`, `git add .`, `git commit -m "Add AI OCR internship project"` and push to your repository).
4. Do not upload a virtual environment or generated output files.

## Internship submission checklist
- [ ] Install Tesseract and Python dependencies.
- [ ] Run `python main.py`.
- [ ] Confirm all four output artifacts are created.
- [ ] Test at least one separate image.
- [ ] Add screenshots of your run/output if your portal asks for evidence.
