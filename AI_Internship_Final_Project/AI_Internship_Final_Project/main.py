"""
AI Internship Final Project — Image Text Recognition (OCR)
Uses OpenCV preprocessing and Tesseract OCR with an 80% confidence filter.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import cv2
import pytesseract
from pytesseract import Output

def preprocess(image):
    """Convert image to a high-contrast binary image for OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    # Otsu thresholding adapts the cutoff to the input image.
    _, binary = cv2.threshold(
        denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return gray, binary

def run_ocr(image_path: str, confidence_threshold: float = 80.0,
            output_dir: str = "outputs"):
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not open image: {path}. Try PNG or JPG.")

    gray, processed = preprocess(image)
    os.makedirs(output_dir, exist_ok=True)
    stem = path.stem
    processed_path = Path(output_dir) / f"{stem}_preprocessed.png"
    annotated_path = Path(output_dir) / f"{stem}_annotated.jpg"
    text_path = Path(output_dir) / f"{stem}_recognized.txt"
    json_path = Path(output_dir) / f"{stem}_results.json"

    cv2.imwrite(str(processed_path), processed)

    data = pytesseract.image_to_data(
        processed, output_type=Output.DICT, config="--oem 3 --psm 6"
    )
    accepted = []
    rejected = []
    annotated = image.copy()

    for i, raw_text in enumerate(data["text"]):
        token = raw_text.strip()
        if not token:
            continue
        try:
            confidence = float(data["conf"][i])
        except (TypeError, ValueError):
            confidence = -1.0

        x, y = int(data["left"][i]), int(data["top"][i])
        w, h = int(data["width"][i]), int(data["height"][i])
        item = {"text": token, "confidence": round(confidence, 2),
                "box": {"x": x, "y": y, "width": w, "height": h}}
        if confidence >= confidence_threshold:
            accepted.append(item)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 180, 0), 2)
            cv2.putText(annotated, f"{token} {confidence:.0f}%",
                        (x, max(18, y-5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 120, 0), 1, cv2.LINE_AA)
        else:
            rejected.append(item)

    cv2.imwrite(str(annotated_path), annotated)
    recognized_text = " ".join(item["text"] for item in accepted)
    text_path.write_text(recognized_text + ("\n" if recognized_text else ""),
                         encoding="utf-8")

    result = {
        "input_image": str(path),
        "confidence_threshold": confidence_threshold,
        "recognized_text": recognized_text,
        "accepted_tokens": accepted,
        "filtered_tokens": rejected,
        "summary": {
            "accepted_count": len(accepted),
            "filtered_count": len(rejected),
            "average_accepted_confidence": round(
                sum(x["confidence"] for x in accepted) / len(accepted), 2
            ) if accepted else None
        },
        "artifacts": {
            "preprocessed_image": str(processed_path),
            "annotated_image": str(annotated_path),
            "text_file": str(text_path)
        }
    }
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n=== AI IMAGE TEXT RECOGNITION RESULTS ===")
    print(f"Input image : {path}")
    print(f"Threshold   : {confidence_threshold:.0f}%")
    print(f"Recognized  : {recognized_text or '[No words passed the confidence filter]'}")
    print(f"Accepted    : {len(accepted)} token(s)")
    print(f"Filtered    : {len(rejected)} token(s)")
    print(f"Annotated   : {annotated_path}")
    print(f"Text output : {text_path}")
    print(f"JSON report : {json_path}")
    return result

def main():
    parser = argparse.ArgumentParser(
        description="OCR project using OpenCV + Tesseract with confidence filtering."
    )
    parser.add_argument("--image", default="samples/sample_text.png",
                        help="Path to an input image (PNG/JPG).")
    parser.add_argument("--threshold", type=float, default=80.0,
                        help="Minimum Tesseract confidence from 0 to 100 (default: 80).")
    parser.add_argument("--output", default="outputs",
                        help="Directory for generated results.")
    parser.add_argument("--tesseract-cmd", default=None,
                        help="Optional path to the Tesseract executable.")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 100:
        parser.error("--threshold must be between 0 and 100")
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd
    try:
        run_ocr(args.image, args.threshold, args.output)
    except pytesseract.TesseractNotFoundError:
        raise SystemExit(
            "Tesseract executable not found. Install Tesseract OCR and ensure it is on PATH; "
            "see README.md for instructions."
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))

if __name__ == "__main__":
    main()
