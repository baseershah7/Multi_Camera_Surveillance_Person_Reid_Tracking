
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import yaml
from ui.multi_camera_gui import MultiCameraGUI

def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    config = load_config()
    gui = MultiCameraGUI(config)
    gui.run()