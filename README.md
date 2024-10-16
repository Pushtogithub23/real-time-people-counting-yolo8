# YOLO-Based Real-time People Tracking and Counting

This repository contains scripts using the YOLO (You Only Look Once) model for real-time object detection and tracking. The focus is on detecting and counting people and vehicles within defined polygon zones. This project utilizes the `ultralytics` YOLO model and `supervision` for annotation and tracking.

## Features

- **Real-time object detection** using YOLO models.
- **Tracking and counting** of people and vehicles crossing predefined polygon zones.
- **Video processing** with annotations for objects inside or outside designated zones.
- Flexible configuration of zones for different counting regions.
- Generates annotated videos that display live counts of tracked objects.

## Prerequisites

- Python 3.8+
- OpenCV
- Numpy
- Ultralytics YOLOv8
- Supervision

Install dependencies:
To install the dependencies, you can run
```bash
pip install -r requirements.txt
```

## Files Overview

### 1. `footfall_counter_1.py`
This script detects and counts people crossing a polygon zone in a video. It tracks people's movement and displays the number of people who crossed the predefined zone.
- **Input**: Video file (`people_in_streets_1.mp4`).
- **Output**: Annotated video (`zone_counter_1.mp4`) showing people detections and counts.
- **Main functionality**:
  - Initializes the YOLO model for detecting people.
  - Defines a polygon zone for counting people crossing the area.
  - Annotates the video with bounding boxes, labels, and the total count of people crossed.

![zone_counter_1](https://github.com/user-attachments/assets/105a9f07-6b20-4c57-bb5d-8b869b8a3c43)


### 2. `people_counter_1.py`
Similar to the `footfall_counter_1.py`, this script counts people in a different polygon zone but on another video. It tracks individuals and displays counts in a different region of the frame.
- **Input**: Video file (`people_in_streets_2.mp4`).
- **Output**: Annotated video (`people_counter_1.mp4`).
- **Main functionality**:
  - Uses YOLO for person detection and counts individuals inside a predefined polygon zone.
  - Annotates bounding boxes, labels, and the total count of people【8†source】.

![people_counter_1](https://github.com/user-attachments/assets/19de88d0-b9e3-453c-859a-7f8f4bd961ae)


### 3. `people_car_counter.py`
This script detects and counts both people and vehicles in a video, using different zones for each. It identifies classes like cars, buses, trucks, and motorbikes, and provides a count for both people and vehicles crossing their respective zones.
- **Input**: Video file (`car_and_people.mp4`).
- **Output**: Annotated video (`zone_counter_3.mp4`).
- **Main functionality**:
  - Detects multiple object classes: people, cars, motorbikes, buses, and trucks.
  - Uses distinct polygon zones to count people and vehicles separately.
  - Displays live counts of people and vehicles on the video【9†source】.

![zone_counter_3](https://github.com/user-attachments/assets/32e7b429-ad83-4c09-afca-6d2cddeccc39)

### 4. `zone_counter_2.py`
This script provides another implementation for people counting in two distinct zones within a video. It applies YOLO for person detection and tracks individuals who cross either of the two polygon zones, maintaining separate counts for each zone.
- **Input**: Video file (`people_in_streets_3.mp4`).
- **Output**: Annotated video (`zone_counter_2.mp4`).
- **Main functionality**:
  - Defines two polygon zones for counting people.
  - Tracks individuals entering each zone and maintains separate counts for both zones.
  - Annotates the video with labels and real-time crossing counts.

![zone_counter_2](https://github.com/user-attachments/assets/2a102ca8-244b-431c-83eb-d960357df89a)

## Usage

To run any of the scripts, simply execute it as follows:

```bash
python footfall_counter_1.py
```

Replace `footfall_counter_1.py` with the desired script name. Ensure the input video paths are correctly configured and the corresponding YOLO model weights are available.

## Customization

You can modify the polygon zones and detection criteria by adjusting the coordinates in each script. The models and detection thresholds can also be customized to suit different object types or video inputs.
