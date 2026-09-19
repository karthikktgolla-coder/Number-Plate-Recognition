import streamlit as st
import cv2
import numpy as np
from PIL import Image

from src.plate_detection import (
    detect_number_plates,
    draw_plate_boxes
)

from src.preprocessing import preprocess_plate
from src.ocr import recognize_plate


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Number Plate Recognition",
    page_icon="🚗",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🚗 Number Plate Recognition")

st.write(
    "Upload a vehicle image to detect and recognize "
    "multiple number plates using Computer Vision and OCR."
)

st.divider()


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📷 Upload a vehicle image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# PROCESS IMAGE
# --------------------------------------------------

if uploaded_file is not None:

    # Read uploaded image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_array = np.array(image)

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------

    st.subheader("📷 Original Image")

    st.image(
        image,
        use_container_width=True
    )


    # --------------------------------------------------
    # DETECT MULTIPLE PLATES
    # --------------------------------------------------

    with st.spinner(
        "🔍 Detecting all number plates..."
    ):

        plates, boxes = detect_number_plates(
            image_bgr
        )


    # --------------------------------------------------
    # NO PLATES FOUND
    # --------------------------------------------------

    if not plates:

        st.error(
            "❌ No number plates could be detected."
        )

        st.info(
            "Try a clear image where the number plates "
            "are visible and facing the camera."
        )


    # --------------------------------------------------
    # PLATES FOUND
    # --------------------------------------------------

    else:

        st.success(
            f"✅ {len(plates)} number plate(s) detected!"
        )


        # --------------------------------------------------
        # IMAGE WITH ALL BOUNDING BOXES
        # --------------------------------------------------

        st.subheader(
            "🔍 All Detected Number Plates"
        )

        result_image = draw_plate_boxes(
            image_bgr,
            boxes
        )

        result_rgb = cv2.cvtColor(
            result_image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            result_rgb,
            caption="All detected number plate regions",
            use_container_width=True
        )


        st.divider()


        # --------------------------------------------------
        # PROCESS EACH PLATE
        # --------------------------------------------------

        for index, plate in enumerate(
            plates,
            start=1
        ):

            st.subheader(
                f"🚘 Number Plate {index}"
            )


            # ------------------------------------------
            # TWO COLUMN LAYOUT
            # ------------------------------------------

            col1, col2 = st.columns(2)


            # ------------------------------------------
            # CROPPED PLATE
            # ------------------------------------------

            with col1:

                st.write(
                    "**Extracted Plate**"
                )

                plate_rgb = cv2.cvtColor(
                    plate,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    plate_rgb,
                    use_container_width=True
                )


            # ------------------------------------------
            # PREPROCESSING
            # ------------------------------------------

            processed_plate = preprocess_plate(
                plate
            )


            with col2:

                st.write(
                    "**Processed Plate**"
                )

                st.image(
                    processed_plate,
                    caption="OCR-ready image",
                    use_container_width=True
                )


            # ------------------------------------------
            # OCR
            # ------------------------------------------

            with st.spinner(
                f"🔤 Reading Plate {index}..."
            ):

                plate_text = recognize_plate(
                    plate
                )


            # ------------------------------------------
            # OCR RESULT
            # ------------------------------------------

            if plate_text:

                st.success(
                    f"🔢 Recognized Number: **{plate_text}**"
                )

            else:

                st.warning(
                    "⚠️ Plate detected, but characters "
                    "could not be recognized."
                )


            st.divider()


# --------------------------------------------------
# HOW IT WORKS
# --------------------------------------------------

with st.expander(
    "🧠 How does this system work?"
):

    st.write(
        """
        **Step 1 — Image Upload**

        The user uploads a vehicle image containing
        one or more vehicles.

        **Step 2 — Multiple Number Plate Detection**

        OpenCV analyzes the image and searches for
        multiple rectangular regions that may represent
        vehicle number plates.

        **Step 3 — Plate Extraction**

        Each detected number plate is cropped
        separately from the original image.

        **Step 4 — Image Preprocessing**

        Each plate is converted to grayscale,
        enlarged, enhanced and thresholded.

        **Step 5 — OCR**

        Tesseract OCR reads the characters from
        each processed number plate.

        **Step 6 — Post-processing**

        The OCR output is converted to uppercase
        and unwanted characters are removed.

        **Step 7 — Final Results**

        The system displays every detected plate
        and its recognized vehicle number.
        """
    )


# --------------------------------------------------
# TECHNOLOGIES
# --------------------------------------------------

with st.expander(
    "📚 Technologies Used"
):

    st.write(
        """
        - Python
        - OpenCV
        - Tesseract OCR
        - Pytesseract
        - NumPy
        - Pillow
        - Streamlit
        - Computer Vision
        - Object Detection
        - Optical Character Recognition (OCR)
        - Image Preprocessing
        """
    )