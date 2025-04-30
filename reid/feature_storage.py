import os
import numpy as np

def save_features(st_reid, config):
    """Save feature embeddings to disk"""
    embeddings_dir = os.path.join(os.getcwd(), config['paths']['embeddings_dir'])
    os.makedirs(embeddings_dir, exist_ok=True)
    
    saved_count = 0
    with st_reid.lock:
        for pid, person in st_reid.tracked_persons.items():
            features = person['features']
            name = person['name']
            color = person['color']
            temp_frames = person['temp_frames']
            file_path = os.path.join(embeddings_dir, f"{pid}.npz")
            np.savez(file_path, features=features, name=name, color=color, temp_frames=temp_frames)
            saved_count += 1
    
    print(f"Saved {saved_count} embeddings to {embeddings_dir}")
    return saved_count

def load_features(st_reid, config):
    """Load feature embeddings from disk"""
    embeddings_dir = os.path.join(os.getcwd(), config['paths']['embeddings_dir'])
    if not os.path.exists(embeddings_dir):
        print("No embeddings directory found.")
        return 0
    
    loaded_count = 0
    loaded_pids = []
    
    with st_reid.lock:
        for file_name in os.listdir(embeddings_dir):
            if not file_name.endswith('.npz'):
                continue
            
            pid_str = os.path.splitext(file_name)[0]
            try:
                pid = int(pid_str)
            except ValueError:
                continue
                
            file_path = os.path.join(embeddings_dir, file_name)
            data = np.load(file_path)
            
            features = data['features']
            name = data['name'].item() if 'name' in data else None
            color = tuple(data['color'].tolist())
            temp_frames = data['temp_frames'].item() if 'temp_frames' in data else 0
            
            features = features[-st_reid.max_gallery_size:]
            
            st_reid.tracked_persons[pid] = {
                'features': features,
                'temp_frames': temp_frames,
                'name': name,
                'color': color
            }
            st_reid.color_map[pid] = color
            loaded_pids.append(pid)
            loaded_count += 1
        
        if loaded_pids:
            max_pid = max(loaded_pids)
            if max_pid >= st_reid.next_id:
                st_reid.next_id = max_pid + 1
                
    print(f"Loaded {loaded_count} embeddings from {embeddings_dir}")
    return loaded_count