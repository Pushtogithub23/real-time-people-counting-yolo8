import cv2 as cv
from ultralytics import YOLO
import numpy as np
import supervision as sv

# Load YOLO model and get class names
model = YOLO("../yolo weights/yolov8n.pt")
# Video input and properties
video_path = "DATA/INPUTS/people_in_streets_3.mp4"
video_info = sv.VideoInfo.from_video_path(video_path)
w, h, fps = (video_info.width, video_info.height, video_info.fps)

# Setup annotators and polygon zone
thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
polygon_points = [
    np.array([[208, 545], [869, 527], [1035, 580], [356, 617]], dtype=np.int32),
    np.array([[1180, 636], [1576, 650], [1780, 854], [1559, 1038], [1134, 997], [976, 822]], dtype=np.int32)
]
zones = [sv.PolygonZone(polygon_point) for polygon_point in polygon_points]

# Initialize tracker, smoother, and crossing tracker
tracker = sv.ByteTrack(frame_rate=fps)

total_counts_1, crossed_ids_1 = [], set()
total_counts_2, crossed_ids_2 = [], set()

colors = sv.ColorPalette.from_hex(["#07f50b", "#07caf5"])


def count_people(ID, cx, y2, zone_id):
    """Count person if the center point (cx, y2) is inside the polygon."""
    if zone_id == 0 and ID not in crossed_ids_1:
        if cv.pointPolygonTest(polygon_points[zone_id], (cx, y2), False) >= 0:
            total_counts_1.append(ID)
            crossed_ids_1.add(ID)
    elif zone_id == 1 and ID not in crossed_ids_2:
        if cv.pointPolygonTest(polygon_points[zone_id], (cx, y2), False) >= 0:
            total_counts_2.append(ID)
            crossed_ids_2.add(ID)


def draw_tracks_and_count(frame, detections):
    """Process detections, annotate based on position inside or outside the polygon."""
    detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]  # Filter for people

    for idx, zone in enumerate(zones):
        mask = zone.trigger(detections)
        zone_annotator = sv.PolygonZoneAnnotator(zone, color=colors.by_idx(idx), text_scale=1.25, text_thickness=2)
        zone_annotator.annotate(frame)
        detections_inside = detections[mask]
        detections_outside = detections[~mask]

        if len(detections_inside) > 0:
            box_annotator = sv.RoundBoxAnnotator(thickness=2, color=colors.by_idx(idx))
            label_annotator = sv.LabelAnnotator(text_scale=0.8, text_thickness=2, color=colors.by_idx(idx),
                                                text_color=sv.Color.BLACK)
            box_annotator.annotate(frame, detections=detections_inside)
            label_annotator.annotate(frame, detections=detections_inside, labels=[f"#{trk_id}" for trk_id in
                                                                                  detections_inside.tracker_id])
        elif len(detections_outside) > 0:
            box_annotator = sv.RoundBoxAnnotator(thickness=2, color=sv.Color.RED)
            label_annotator = sv.LabelAnnotator(text_scale=0.8, text_thickness=2, color=sv.Color.RED,
                                                text_color=sv.Color.BLACK)
            box_annotator.annotate(frame, detections=detections_outside)
            label_annotator.annotate(frame, detections=detections_outside, labels=[f"#{trk_id}" for trk_id in
                                                                                   detections_outside.tracker_id])

    # Draw circles at the bottom center of each detection
    for track_id, bottom_center in zip(detections.tracker_id,
                                       detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)):
        cx, y2 = map(int, bottom_center)
        cv.circle(frame, (cx, y2), 4, (0, 255, 255), cv.FILLED)
        count_people(track_id, cx, y2, 0)
        count_people(track_id, cx, y2, 1)

    cv.rectangle(frame, (0, 0), (400, 120), (255, 255, 255), cv.FILLED)
    count_labels = [f"ZONE_1: {len(total_counts_1)}", f"ZONE_2: {len(total_counts_2)}"]
    zone_colors = [(11, 245, 7), (245, 202, 7)]
    for i, (labels, color) in enumerate(zip(count_labels, zone_colors)):
        cv.putText(frame, labels, (10, 50 + i * 50), cv.FONT_HERSHEY_PLAIN, 3, color, 3)


# Video loading and display
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception("Error: couldn't open the video!")
filename = "DATA/OUTPUTS/zone_counter_2.mp4"
out = cv.VideoWriter(filename, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO model detection, tracking, and smoothing
    results = model(frame)[0]
    detections = tracker.update_with_detections(sv.Detections.from_ultralytics(results))

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
