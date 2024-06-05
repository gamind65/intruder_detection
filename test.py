from module.detect import *
import json

def main():
    with open("config.json", "r") as f:
        config = json.load(f)
    
    yo = YoloDetect(model_path=config['model_path'], poly=config['poly'], conf_thresh=config['conf_thresh'], cam_idx=config['cam_idx'])
    yo.detect(0)
    
if __name__ == "__main__":
    main()