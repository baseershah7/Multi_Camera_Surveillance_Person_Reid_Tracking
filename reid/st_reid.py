import torch
import numpy as np
import cv2
from PIL import Image
import threading
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from torchreid.reid.models import build_model
from torchreid.reid.data.transforms import build_transforms

class STReID:
    def __init__(self, config):
        reid_config = config['reid']
        self.alpha = reid_config['alpha']
        self.threshold = reid_config['threshold']
        self.temporal_frames = reid_config['temporal_frames']
        self.max_gallery_size = reid_config['max_gallery_size']
        self.lock = threading.Lock()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_name = config['models']['reid_gpu'] if torch.cuda.is_available() else config['models']['reid_cpu']
        num_classes = config['models']['reid_num_classes']
        
        self.model = build_model(
            name=model_name,
            num_classes=num_classes,
            pretrained=True
        )
        
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)
            
        self.model.eval()
        
        self.transform = build_transforms(
            height=reid_config['height'],
            width=reid_config['width'],
            norm_mean=reid_config['norm_mean'],
            norm_std=reid_config['norm_std'],
        )[1]

        self.tracked_persons = {}
        self.next_id = 0
        self.color_map = {}

    def extract_features(self, bgr_imgs):
        batch = []
        for img in bgr_imgs:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_t = self.transform(img_pil).unsqueeze(0)
            batch.append(img_t)
        
        with torch.no_grad(), self.lock:
            batch_t = torch.cat(batch).to(self.device)
            feats = self.model(batch_t).cpu().numpy()
        
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
        return feats / norms

    def match_batch(self, feats):
        with self.lock:
            if not self.tracked_persons:
                return [self._create_id(f) for f in feats]

            pids = list(self.tracked_persons.keys())
            gallery_feats = np.array([p['features'].mean(0) for p in self.tracked_persons.values()])
            
            cos_dists = distance.cdist(feats, gallery_feats, 'cosine')
            euc_dists = distance.cdist(feats, gallery_feats, 'euclidean')
            cost_matrix = self.alpha * cos_dists + (1 - self.alpha) * euc_dists

            row_idx, col_idx = linear_sum_assignment(cost_matrix)
            assignments = [None] * len(feats)
            
            for r, c in zip(row_idx, col_idx):
                if cost_matrix[r, c] < self.threshold:
                    pid = pids[c]
                    assignments[r] = pid
                    self._update_gallery(pid, feats[r])
                else:
                    assignments[r] = self._create_id(feats[r])
            
            for i in range(len(assignments)):
                if assignments[i] is None:
                    assignments[i] = self._create_id(feats[i])
            
            return assignments

    def _create_id(self, feat):
        pid = self.next_id
        self.next_id += 1
        color = tuple(np.random.randint(0, 255, 3).tolist())
        self.color_map[pid] = color
        self.tracked_persons[pid] = {
            'features': np.array([feat]),
            'temp_frames': 0,
            'name': None,
            'color': color
        }
        print(f"Created ID {pid} with color {color}")
        return pid

    def _update_gallery(self, pid, feat):
        entry = self.tracked_persons[pid]
        features = entry['features']
        entry['features'] = np.vstack((features[-(self.max_gallery_size - 1):], feat))
        entry['temp_frames'] = min(entry['temp_frames'] + 1, self.temporal_frames)

    def set_threshold(self, val):
        with self.lock:
            self.threshold = float(val)

    def rename_id(self, pid, new_name):
        with self.lock:
            if pid in self.tracked_persons:
                self.tracked_persons[pid]['name'] = new_name
                print(f"Renamed ID {pid} to {new_name}")