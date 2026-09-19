# Number Plate Recognition

A computer vision application that automatically detects vehicle number plates from images and recognizes the characters using Optical Character Recognition (OCR).

## 📌 Project Overview

The **Number Plate Recognition** system uses **YOLO** for number plate detection and **Tesseract OCR** for reading the characters on the detected plate.

The application provides a simple **Streamlit web interface** where users can upload an image and view the detected number plates along with the recognized plate numbers.

## 🚀 Features

* Upload vehicle images through a web interface
* Detect number plates using YOLO
* Detect multiple number plates in a single image
* Crop detected number plates automatically
* Preprocess images for better OCR accuracy
* Recognize plate characters using Tesseract OCR
* Display detected plates with bounding boxes
* Display recognized number plate text
* Simple and user-friendly Streamlit interface

## 🛠️ Technologies Used

* **Python**
* **YOLO (Ultralytics)**
* **OpenCV**
* **Tesseract OCR**
* **Pytesseract**
* **NumPy**
* **Pillow**
* **Streamlit**

## 🔄 System Workflow

```text
Input Vehicle Image
        ↓
YOLO Number Plate Detection
        ↓
Number Plate Cropping
        ↓
Image Preprocessing
        ↓
Tesseract OCR
        ↓
Text Cleaning
        ↓
Recognized Number Plate
```

## 📂 Project Structure

```text
Number Plate Recognition/
│
├── data/
│
├── test_images/
│
├── models/
│   └── license-plate-finetune-v1n.pt
│
├── src/
│   ├── plate_detection.py
│   ├── ocr.py
│   └── preprocessing.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/karthikktgolla-coder/Number-Plate-Recognition.git
cd Number-Plate-Recognition
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

Install Tesseract OCR on your system and make sure the executable path is correctly configured.

The application currently uses:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

## ▶️ Run the Application

Start the Streamlit application using:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

Upload a vehicle image to detect and recognize the number plate.

## 🎯 AI Concepts Used

### Computer Vision

OpenCV is used for image processing and preparation of number plate images.

### Object Detection

YOLO is used to locate number plates in vehicle images.

### Optical Character Recognition

Tesseract OCR is used to convert the characters present on the detected number plate into text.

### Image Preprocessing

The detected plate is enlarged, converted to grayscale, enhanced, and thresholded to improve OCR recognition.

## 🔮 Future Enhancements

* Real-time number plate recognition using a camera
* Support for video input
* Improved OCR accuracy using deep learning OCR models
* Automatic vehicle information storage
* Database integration
* Entry and exit tracking system
* Integration with parking management systems
* Support for additional number plate formats

## 📜 Project Type

**B.Tech Minor Project**

## 👨‍💻 Author

**Karthik Golla**

## 📄 Project 

Number Plate Recognition
