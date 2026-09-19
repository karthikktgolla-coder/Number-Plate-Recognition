from ultralytics import YOLO
import cv2
import os


# --------------------------------------------------
# YOLO LICENSE PLATE MODEL
# --------------------------------------------------

MODEL_PATH = os.path.join(
    "models",
    "license-plate-finetune-v1n.pt"
)

model = YOLO(MODEL_PATH)


# --------------------------------------------------
# DETECT NUMBER PLATES
# --------------------------------------------------

def detect_number_plates(image):
    """
    Detect only vehicle number plates using YOLO.

    Returns:
        plates: cropped number plate images
        boxes: bounding boxes
    """

    results = model.predict(
        source=image,
        conf=0.35,
        imgsz=1280,
        iou=0.45,
        max_det=20,
        verbose=False
    )

    plates = []
    boxes = []

    if not results:
        return plates, boxes

    result = results[0]

    if result.boxes is None:
        return plates, boxes

    image_height, image_width = image.shape[:2]

    for box in result.boxes:

        # Get confidence
        confidence = float(
            box.conf[0]
        )

        # Ignore weak detections
        if confidence < 0.35:
            continue

        # Get bounding box coordinates
        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        # Keep coordinates inside image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            continue

        # Ignore extremely tiny detections
        if width < 20 or height < 8:
            continue

        # Crop the detected number plate
        plate = image[
            y1:y2,
            x1:x2
        ]

        if plate.size == 0:
            continue

        boxes.append(
            (
                x1,
                y1,
                width,
                height
            )
        )

        plates.append(plate)

    # Sort plates from left to right
    combined = list(
        zip(boxes, plates)
    )

    combined.sort(
        key=lambda item: (
            item[0][1],
            item[0][0]
        )
    )

    boxes = [
        item[0]
        for item in combined
    ]

    plates = [
        item[1]
        for item in combined
    ]

    return plates, boxes


# --------------------------------------------------
# DRAW DETECTION BOXES
# --------------------------------------------------

def draw_plate_boxes(image, boxes):
    """
    Draw boxes only around detected number plates.
    """

    result = image.copy()

    for index, box in enumerate(
        boxes,
        start=1
    ):

        x, y, width, height = box

        # Draw number plate box
        cv2.rectangle(
            result,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            3
        )

        # Label
        cv2.putText(
            result,
            f"Number Plate {index}",
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return result