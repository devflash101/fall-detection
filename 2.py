from ultralytics import YOLO
import cv2
from collections import deque
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

def detect_from_video(filename):

    # Load the model
    model = YOLO("fall_detection.pt")

    # Initialize the DeepSort tracker
    tracker = DeepSort(max_age=50)

    # Open the video file
    # video = cv2.VideoCapture("1.mp4")
    video = cv2.VideoCapture(filename)


    # Get video properties
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a buffer to store recent frames
    frame_buffer = deque(maxlen=fps*2)  # 2 seconds of frames

    # Initialize variables
    clip_counter = 0
    saving_clip = False
    new_object_found = False
    previous_track_ids = set()

    while video.isOpened():
        success, frame = video.read()
        
        if not success:
            break

        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        # annotated_frame = results[0].plot()
        annotated_frame = frame.copy()

        # Add annotated frame to buffer
        frame_buffer.append(annotated_frame)

        # child_without_hat_detected = False

        detect = []
        # Process detection results
        for r in results:
            for box in r.boxes:
                # Get the predicted class label and confidence
                class_id = int(box.cls[0])
                label = model.names[class_id]
                conf = float(box.conf[0])
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()



                # Check if it's a child without a hat
                # if label.startswith('n '):
                if label == 'f':

                    print(f"Child without hat detected! Confidence: {conf:.2f}")
                    # Draw rectangle on the frame - only for the children without hat
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    print('bbbbbbbbbbbbbbbbbbbbbbbbbbbb', class_id)
                detect.append([[x1, y1, x2 - x1, y2 - y1], conf, label])
                    
        tracks = tracker.update_tracks(detect, frame=frame)
        
        current_track_ids = set()  
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            current_track_ids.add(track_id)
        
            if track_id not in previous_track_ids:
                if track.det_class == 'f':
                    new_object_found = True
                print(f"New object detected! Track ID: {track_id}")
            
        previous_track_ids = current_track_ids

        # Start saving clip if threshold is reached
        if  new_object_found and not saving_clip:
            saving_clip = True
            new_object_found = False
            clip_counter += 1
            # print('464362357257282546272546734767373')
            
            # Create video writer
            out = cv2.VideoWriter(f'clip_{clip_counter}.mp4', 
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                fps, (width, height))
            
            # # Write buffered frames
            # for buffered_frame in frame_buffer:
            #     out.write(buffered_frame)
            
            # Reset frame counter
            frame_count = 0

        # If we're in the process of saving a clip, write the frame
        if saving_clip:
            out.write(annotated_frame)
            frame_count += 1

            # If we've written 2 seconds worth of frames, stop saving
            if frame_count >= fps * 2:
                saving_clip = False
                out.release()
        
        # Display the frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close the display window
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_from_video('1.mp4')