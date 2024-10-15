import cv2 as cv
from ultralytics import YOLO
import numpy as np
import supervision as sv

# Load YOLO model and get class names
model = YOLO("yolov10x.pt")
# Video input and properties
video_path = "DATA/INPUTS/car_and_people.mp4"
video_info = sv.VideoInfo.from_video_path(video_path)
w, h, fps = (video_info.width, video_info.height, video_info.fps)

# Define zones for people and cars
people_zone_points = [
    np.array([[305, 686], [613, 682], [613, 682], [708, 740], [260, 752]], dtype=np.int32),
    np.array([[403, 836], [886, 830], [1163, 964], [557, 987]], dtype=np.int32)
]

car_zone_points = np.array([[717, 730], [1291, 587], [1920, 648], [1920, 1080], [1428, 1080]], dtype=np.int32)
people_zones = [sv.PolygonZone(zone) for zone in people_zone_points]
car_zone = sv.PolygonZone(car_zone_points)

# Initialize tracker and smoother
tracker = sv.ByteTrack(frame_rate=fps)
smoother = sv.DetectionsSmoother()

# Initialize counters
total_people_count, crossed_people_ids = [], set()
total_car_count, crossed_car_ids = [], set()
class_names = model.names
classes_to_track = ['person', 'car', 'motorbike', 'bus', 'truck']
tracked_classes = [cls_id for cls_id, cls_name in model.names.items() if cls_name in classes_to_track]


def count_people(ID, cx, y2):
    """Count people if the center point (cx, y2) is inside any people zone."""
    for zone_points in people_zone_points:
        if cv.pointPolygonTest(zone_points, (cx, y2), False) >= 0 and ID not in crossed_people_ids:
            total_people_count.append(ID)
            crossed_people_ids.add(ID)


def count_vehicles(ID, cx, y2):
    """Count vehicles if the center point (cx, y2) is inside the car zone."""
    if cv.pointPolygonTest(car_zone_points, (cx, y2), False) >= 0 and ID not in crossed_car_ids:
        total_car_count.append(ID)
        crossed_car_ids.add(ID)


def annotate(frame, detections, labels):
    """Helper function to annotate boxes, labels, and traces."""
    box_annotator = sv.RoundBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.8, text_thickness=2,
                                        text_position=sv.Position.TOP_CENTER,
                                        )

    box_annotator.annotate(frame, detections=detections)
    label_annotator.annotate(frame, detections=detections, labels=labels)


def draw_tracks_and_count(frame, detections):
    """Process detections and annotate zones."""
    detections = detections[(np.isin(detections.class_id, tracked_classes)) & (detections.confidence > 0.5)]

    # Annotate and count for people zones
    for zone, color in zip(people_zones, [sv.Color.RED, sv.Color.GREEN]):
        zone_annotator = sv.PolygonZoneAnnotator(zone, thickness=4, color=color, text_scale=1, text_thickness=2)
        zone.trigger(detections)
        zone_annotator.annotate(frame)

    # Annotate and count for car zone
    car_zone_annotator = sv.PolygonZoneAnnotator(car_zone, thickness=4, color=sv.Color.BLUE, text_scale=2,
                                                 text_thickness=2)
    car_zone.trigger(detections)
    car_zone_annotator.annotate(frame)

    labels = [f"{class_names[cls_id]} #{tid}" for cls_id, tid in zip(detections.class_id, detections.tracker_id)]
    annotate(frame, detections, labels)

    # Draw circles at the bottom center of each detection and count people/vehicles
    for track_id, bottom_center in zip(detections.tracker_id,
                                       detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)):
        cx, y2 = map(int, bottom_center)
        cv.circle(frame, (cx, y2), 4, (0, 255, 255), cv.FILLED)
        count_people(track_id, cx, y2)
        count_vehicles(track_id, cx, y2)

    # Display the crossing counts
    count_labels = [f"People crossed: {len(total_people_count)}", f"Cars crossed: {len(total_car_count)}"]
    count_colors = [(4, 165, 10), (180, 4, 6)]
    cv.rectangle(frame, (0,0), (500, 120), (255, 255, 255), cv.FILLED)
    for i, (label, color) in enumerate(zip(count_labels, count_colors)):
        cv.putText(frame, label, (15, 40+i*50), cv.FONT_HERSHEY_COMPLEX, 1.5, color, 2)

# Video loading and display
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception("Error: couldn't open the video!")

filename = "DATA/OUTPUTS/zone_counter_3.mp4"
out = cv.VideoWriter(filename, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = smoother.update_with_detections(
                 tracker.update_with_detections(sv.Detections.from_ultralytics(results)))

    if detections.tracker_id is not None:
        draw_tracks_and_count(frame, detections)

    # Write frames to save the video
    out.write(frame)
    # Display the video
    cv.imshow("Video", frame)
    # Exit on pressing 'p'
    if cv.waitKey(1) & 0xff==ord('p'):
        break

# Release Resources
cap.release()
out.release()
cv.destroyAllWindows()
