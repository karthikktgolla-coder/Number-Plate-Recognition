import cv2


def preprocess_plate(image):
    """
    Preprocess a number plate image for OCR.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Improve contrast using CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Convert to binary image
    threshold = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Resize for better OCR
    height, width = threshold.shape

    resized = cv2.resize(
        threshold,
        (width * 2, height * 2),
        interpolation=cv2.INTER_CUBIC
    )

    return resized