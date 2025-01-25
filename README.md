# License Plate Detection with Object Tracking and OCR
- This project uses YOLOv8 for object detection, OCSORT for tracking, and EasyOCR/PaddleOCR to read license plate text. It also includes a GUI for EasyOCR and PaddleOCR. If your environment supports CUDA, GPU acceleration is enabled.

## Features
- YOLOv8: Used for detecting license plates in the video feed.
- OCSORT: Object tracking to maintain identity across frames.
- EasyOCR: Optical character recognition (OCR) to extract text from detected license plates.
- PaddleOCR: An alternative OCR tool to read text from license plates.
- GUI Support: Integrated GUI support for EasyOCR and PaddleOCR for enhanced usability.


https://github.com/user-attachments/assets/0019b1e4-5ecc-4931-8479-2f240debec75



https://github.com/user-attachments/assets/90b28089-69e9-4126-a8b5-96228ba2fd48

You can download the sample video [here](https://drive.google.com/file/d/18r5hAev63ai_pVxm5DnxzSrTbQn2OY3s/view?usp=drive_link)
## Requirements
Clone the Repository:
```bash
git clone https://github.com/EiPhyuSinn/LicensePlateDetection.git
```

## Create a virtual environment to avoid package conflicts:
  ```bash
  python3 -m venv testing_env
  source testing_env/bin/activate  # For macOS/Linux
  testing_env\Scripts\activate     # For Windows
```


## Install Dependencies:

- Ensure you have Python 3.8+ installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

## Download Weights:

- Download the pre-trained weights for YOLOv8 [here](https://drive.google.com/file/d/1Yh-1TpgX0eWAZiRRcnoZvzhf2qqyqy9b/view?usp=drive_link)
- Once downloaded, place the weights in the appropriate folder as specified in the code.




## How to Run
## Running the GUI for PaddleOCR:
- If you want to use PaddleOCR for license plate text extraction, you can run the following command:
```bash
python3 src/paddle_gui.py
```

## Running the GUI for EasyOCR:
- If you want to use EasyOCR for license plate text extraction, you can run the following command:
```bash
python3 src/easy_gui.py
```
