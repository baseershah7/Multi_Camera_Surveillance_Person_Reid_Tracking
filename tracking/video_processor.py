import os
import cv2
import torch
import numpy as np
import time
import queue
import threading
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from reid.utils import draw_person_box

class VideoProcessor:
    def __init__(self, video_path, st_reid, output_path, frame_queue, vid_idx, config):
        self.video_path = video_path
        self.st_reid = st_reid
        self.output_path = output_path
        self.frame_queue = frame_queue
        self.vid_idx = vid_idx
        self.config = config
        self.processing_lock = threading.Lock()
        self.running = True
        self.writer = None
        self.cap = None

        # Load YOLO model
        self.yolo = YOLO(config['models']['yolo'])
        if torch.cuda.is_available():
            self.yolo.model.to(st_reid.device)
        self.yolo.classes = [config['detection']['person_class_id']]
        
        # Initialize DeepSort tracker
        tracker_config = config['tracker']
        self.tracker = DeepSort(
            max_age=tracker_config['max_age'],
            n_init=tracker_config['n_init'],
            max_cosine_distance=tracker_config['max_cosine_distance'],
            nn_budget=tracker_config['nn_budget']
        )

    def process_frame(self, frame):
        with self.processing_lock:
            try:
                # Run YOLO detection
                results = self.yolo(frame, verbose=False)[0]
                detections = []
                
                # Process detections
                conf_threshold = self.config['detection']['confidence_threshold']
                person_class_id = self.config['detection']['person_class_id']
                
                for box in results.boxes:
                    if box.conf.item() > conf_threshold and int(box.cls.item()) == person_class_id:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
                        detections.append(([x1, y1, x2 - x1, y2 - y1], box.conf.item(), None))

                # Update tracker
                tracks = self.tracker.update_tracks(detections, frame=frame)
                bboxes = []
                crops = []
                
                # Process confirmed tracks
                for trk in tracks:
                    if trk.is_confirmed() and trk.time_since_update <= 1:
                        l, t, r, b = map(int, trk.to_ltrb())
                        if (r - l) > 10 and (b - t) > 10:
                            bboxes.append((l, t, r, b))
                            crops.append(frame[t:b, l:r])

                # Extract features and match with gallery
                if crops:
                    feats = self.st_reid.extract_features(crops)
                    assignments = self.st_reid.match_batch(feats)
                    
                    # Draw boxes and labels
                    for idx, (bbox, pid) in enumerate(zip(bboxes, assignments)):
                        try:
                            person_data = self.st_reid.tracked_persons[pid]
                            frame = draw_person_box(frame, bbox, pid, person_data)
                        except KeyError:
                            print(f"Temporary tracking mismatch for PID {pid}")
                            continue
            
                return frame
            
            except Exception as e:
                print(f"Processing error: {e}")
                return frame
                
    def initialize_video(self):
        # Open video capture
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Cannot open video: {self.video_path}")
            return False
            
        # Get video properties
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Use a more reliable codec - H.264 for better compatibility
        # On Windows, use 'avc1' or H264 codec
        if os.name == 'nt':  # Windows
            self.writer = cv2.VideoWriter(
                self.output_path,
                cv2.VideoWriter_fourcc(*'avc1'),  # H.264 codec
                fps,
                (w, h)
            )
        else:  # Linux/Mac
            self.writer = cv2.VideoWriter(
                self.output_path,
                cv2.VideoWriter_fourcc(*'avc1'),  # H.264 codec 
                fps,
                (w, h)
            )
            
        # Check if writer was successfully created
        if not self.writer.isOpened():
            # Fall back to mp4v if avc1 fails
            self.writer = cv2.VideoWriter(
                self.output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (w, h)
            )
            
        return self.writer.isOpened() and self.cap.isOpened()

    def release_resources(self):
        # Properly release resources
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def run(self):
        try:
            if not self.initialize_video():
                return
                
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(0.001, 1/fps) if fps > 0 else 0.033

            # Main processing loop
            while self.running and self.cap.isOpened():
                start_time = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    break

                processed = self.process_frame(frame)
                self.writer.write(processed)
                
                # Send frame to UI
                try:
                    self.frame_queue.put_nowait((self.vid_idx, processed))
                except queue.Full:
                    pass
                
                # Control frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                time.sleep(sleep_time)

        except Exception as e:
            print(f"Processor {self.vid_idx} crashed: {e}")
        finally:
            # Make sure resources are always released
            self.release_resources()
            print(f"Processor {self.vid_idx} exited")

    def stop(self):
        self.running = False
        # Ensure resources are released when stopped
        self.release_resources()