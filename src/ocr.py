import cv2
import pytesseract
import re


# Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def preprocess_for_ocr(plate_image):
    """
    Create multiple processed versions of a number plate
    to improve OCR recognition.
    """

    gray = cv2.cvtColor(
        plate_image,
        cv2.COLOR_BGR2GRAY
    )

    # Enlarge the plate
    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Reduce noise
    gray = cv2.bilateralFilter(
        gray,
        9,
        75,
        75
    )

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Method 1: OTSU threshold
    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Method 2: Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9
    )

    # Method 3: Inverted OTSU
    _, inverted = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return [
        otsu,
        adaptive,
        inverted
    ]


def extract_text(image):
    """
    Run Tesseract OCR on an image.
    """

    config = (
        "--oem 3 "
        "--psm 7 "
        "-c tessedit_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    text = pytesseract.image_to_string(
        image,
        config=config
    )

    return clean_plate_text(text)


def clean_plate_text(text):
    """
    Clean OCR output.
    """

    text = text.upper()

    # Remove spaces and special characters
    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


def recognize_plate(plate_image):
    """
    Recognize number plate characters using
    multiple preprocessing techniques.
    """

    if plate_image is None:
        return ""

    processed_images = preprocess_for_ocr(
        plate_image
    )

    results = []

    for processed_image in processed_images:

        text = extract_text(
            processed_image
        )

        if text:
            results.append(text)

    if not results:
        return ""

    # Prefer the longest reasonable OCR result
    results.sort(
        key=len,
        reverse=True
    )

    return results[0]