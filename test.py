from utils import YoloDetect
import json

def main():
    with open("config.json", "r") as f:
        config = json.load(f)
    
    cap = cv2.VideoCapture(config['cam_idx'])
    yo = YoloDetect(model_path=config['model_path'], poly=config['poly'], video_cap=cap, conf_thresh=config['conf_thresh'])
    yo.detect()
    
if __name__ == "__main__":
    main()