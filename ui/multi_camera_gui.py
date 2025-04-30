import os
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
import threading
from PIL import Image, ImageTk
import queue
import time
from reid.st_reid import STReID
from reid.feature_storage import save_features, load_features
from tracking.video_processor import VideoProcessor

class MultiCameraGUI:
    def __init__(self, config):
        self.config = config
        self.root = tk.Tk()
        self.root.title("Multi-Cam ReID")
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        self._setup_controls()
        self._init_video_panel()
        self.running = False
        self.processors = []
        self.frame_queues = {}
        self.video_labels = {}
        self.update_thread = None
        self.input_folder = None
        self.output_folder = None

    def _setup_controls(self):
        self.ctrl_panel = ttk.Frame(self.main_container)
        self.ctrl_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        controls = [
            ("Input Folder", self.sel_input),
            ("Output Folder", self.sel_output),
            ("Start", self.start_processing),
            ("Stop", self.stop_processing),
            ("Rename ID", self.rename_id),
            ("Save Features", self.save_features),
            ("Load Features", self.load_features),
            ("Quit", self.quit_app)
        ]
        
        for text, cmd in controls:
            ttk.Button(self.ctrl_panel, text=text, command=cmd).pack(pady=3, fill=tk.X)
        
        ttk.Label(self.ctrl_panel, text="Threshold").pack(pady=5)
        self.thr_scale = tk.Scale(
            self.ctrl_panel, from_=0.0, to=1.0, resolution=0.01,
            orient=tk.HORIZONTAL, command=self.on_thr_change
        )
        self.thr_scale.set(self.config['reid']['threshold'])
        self.thr_scale.pack()

    def _init_video_panel(self):
        self.video_panel = ttk.Frame(self.main_container)
        self.video_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def create_video_display(self, num_streams):
        for widget in self.video_panel.winfo_children():
            widget.destroy()
        
        cols = min(3, num_streams)
        rows = (num_streams + cols - 1) // cols
        self.video_labels = {}
        self.frame_queues = {}
        
        for i in range(num_streams):
            row = i // cols
            col = i % cols
            frame_container = ttk.Frame(self.video_panel)
            frame_container.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            label = ttk.Label(frame_container)
            label.pack()
            self.video_labels[i] = label
            self.frame_queues[i] = queue.Queue(maxsize=10)
        
        for r in range(rows):
            self.video_panel.rowconfigure(r, weight=1)
        for c in range(cols):
            self.video_panel.columnconfigure(c, weight=1)

    def update_frames(self):
        update_interval = 0.033
        last_update = time.time()
        max_width = self.config['video']['display']['max_width']
        max_height = self.config['video']['display']['max_height']
        
        while self.running:
            current_time = time.time()
            if current_time - last_update >= update_interval:
                for idx, label in self.video_labels.items():
                    frame = None
                    try:
                        while True:
                            _, frame = self.frame_queues[idx].get_nowait()
                    except queue.Empty:
                        pass
                    
                    if frame is not None:
                        frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w = frame_disp.shape[:2]
                        ratio = min(max_width / w, max_height / h)
                        new_size = (int(w * ratio), int(h * ratio))
                        frame_disp = cv2.resize(frame_disp, new_size)
                        img = ImageTk.PhotoImage(Image.fromarray(frame_disp))
                        label.config(image=img)
                        label.image = img
                last_update = current_time
                self.root.update()
            else:
                time.sleep(0.001)

    def on_thr_change(self, val):
        new_thr = float(val)
        if self.processors:
            self.processors[0].st_reid.set_threshold(new_thr)
        print(f"Threshold updated to: {new_thr:.2f}")

    def sel_input(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_folder = folder

    def sel_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder

    def rename_id(self):
        if not self.processors:
            print("No active video processing")
            return
    
        pid_str = simpledialog.askstring("Rename ID", "Enter numeric ID:")
        if not pid_str:
            return
    
        try:
            pid = int(pid_str)
        except ValueError:
            print("Invalid ID format")
            return
    
        new_name = simpledialog.askstring("Rename ID", "Enter new name:")
        if not new_name:
            return
    
        try:
            self.processors[0].st_reid.rename_id(pid, new_name)
            print(f"Successfully renamed ID {pid} to {new_name}")
        except KeyError:
            print(f"ID {pid} does not exist in current tracking")

    def save_features(self):
        if not self.processors:
            print("No active processing to save features.")
            return
        st_reid = self.processors[0].st_reid
        save_features(st_reid, self.config)

    def load_features(self):
        if not self.processors:
            print("Start processing first to load features.")
            return
        st_reid = self.processors[0].st_reid
        load_features(st_reid, self.config)

    def start_processing(self):
        if not (self.input_folder and self.output_folder):
            return
        
        formats = self.config['video']['supported_formats']
        max_videos = self.config['video']['max_videos']
        
        videos = [f for f in os.listdir(self.input_folder) 
                  if any(f.lower().endswith(ext) for ext in formats)][:max_videos]
        
        if not videos:
            return
        
        self.create_video_display(len(videos))
        self.running = True
        
        st_reid = STReID(self.config)
        self.processors = []
        
        for i, vf in enumerate(videos):
            in_path = os.path.join(self.input_folder, vf)
            out_path = os.path.join(self.output_folder, f"out_{i}.mp4")
            proc = VideoProcessor(in_path, st_reid, out_path, self.frame_queues[i], i, self.config)
            self.processors.append(proc)
            threading.Thread(target=proc.run, daemon=True).start()
        
        self.update_thread = threading.Thread(target=self.update_frames, daemon=True)
        self.update_thread.start()

    def stop_processing(self):
        self.running = False
        for proc in self.processors:
            proc.stop()
        if self.update_thread:
            self.update_thread.join(timeout=5)
        self.processors = []
        # Add a small delay to ensure all resources are released
        time.sleep(0.5)

    def quit_app(self):
        self.stop_processing()
        self.root.destroy()

    def run(self):
        self.root.mainloop()