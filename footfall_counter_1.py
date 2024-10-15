import cv2 as cv
from ultralytics import YOLO
import numpy as np
import supervision as sv

# Load YOLO model and get class names
model = YOLO("../yolo weights/yolov8n.pt") # replace this with your model's path
# Video input and properties
video_path = "DATA/INPUTS/people_in_streets_1.mp4"
video_info = sv.VideoInfo.from_video_path(video_path)
w, h, fps = (video_info.width, video_info.height, video_info.fps)

# Setup annotators and polygon zone
thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
polygon_points = np.array([[451, 756], [734, 929], [1382, 870], [985, 724]], dtype=np.int32)
zone = sv.PolygonZone(polygon_points)

# Initialize tracker, smoother, and crossing tracker
tracker = sv.ByteTrack(frame_rate=video_info.fps)
smoother = sv.DetectionsSmoother()

total_counts, crossed_ids = [], set()


def count_people(ID, cx, y2):
    """Count person if the center point (cx, y2) is inside the polygon."""
    if cv.pointPolygonTest(polygon_points, (cx, y2), False) >= 0 and ID not in crossed_ids:
        total_counts.append(ID)
        crossed_ids.add(ID)


def annotate(frame, detections, color, labels):
    """Helper function to annotate boxes, labels, and traces."""
    box_annotator = sv.RoundBoxAnnotator(color=color, thickness=thickness)
    label_annotator = sv.LabelAnnotator(color=color, text_scale=text_scale, 
                                        text_thickness=thickness,
                                        text_position=sv.Position.TOP_CENTER)

    box_annotator.annotate(frame, detections=detections)
    label_annotator.annotate(frame, detections=detections, labels=labels)


def draw_tracks_and_count(frame, detections):
    """Process detections, annotate based on position inside or outside the polygon."""
    detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]  # Filter for people
    mask_inside = zone.trigger(detections)  # Mask for people inside the polygon

    # Get inside and outside detections
    detections_inside, detections_outside = detections[mask_inside], detections[~mask_inside]

    # Annotate inside detections in green, outside in red
    if len(detections_inside) > 0:
        labels = [f"#{tid}" for tid in detections_inside.tracker_id]
        annotate(frame, detections_inside, sv.Color.GREEN, labels)
    if len(detections_outside) > 0:
        labels = [f"#{tid}" for tid in detections_outside.tracker_id]
        annotate(frame, detections_outside, sv.Color.RED, labels)

    # Annotate polygon zone, color depending on detections inside
    zone_annotator = sv.PolygonZoneAnnotator(zone, 
                                             color=sv.Color.GREEN if len(detections_inside) else sv.Color.RED,
                                             text_scale=2, text_thickness=2)
    zone_annotator.annotate(frame)

    # Draw circles at the bottom center of each detection
    for track_id, bottom_center in zip(detections.tracker_id,
                                       detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)):
        cx, y2 = map(int, bottom_center)
        cv.circle(frame, (cx, y2), 4, (0, 255, 255), cv.FILLED)
        count_people(track_id, cx, y2)

    # Display the crossing count
    sv.draw_text(frame, f"People crossed: {len(total_counts)}", sv.Point(x=200, y=40), sv.Color.BLACK, 1.25,
                 2, background_color=sv.Color.WHITE)


# Video loading and display
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception("Error: couldn't open the video!")

filename = "DATA/OUTPUTS/zone_counter_1.mp4"
out = cv.VideoWriter(filename, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO model detection, tracking, and smoothing
    results = model(frame)[0]
    detections = smoother.update_with_detections(
                 tracker.update_with_detections(sv.Detections.from_ultralytics(results)))

    # If there are any tracker IDs, process detections
    if detections.tracker_id is not None:
        draw_tracks_and_count(frame, detections)

    # Write frames to save the video
    out.write(frame)
    # Display the resized frame
    resized_frame = cv.resize(frame, (int(0.55 * frame.shape[1]), int(0.55 * frame.shape[0])))
    cv.imshow("Video", resized_frame)

    # Exit on pressing 'p'
    if cv.waitKey(1) & 0xff == ord('p'):
        break

# Release resources
cap.release()
out.release()
cv.destroyAllWindows()
