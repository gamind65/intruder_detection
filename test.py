from module.detect import *
import json

def test(frame, yo, mode=0):
        if mode == 0: yo.track(frame)
        else: yo.detect(frame)

def main():
    # setup 
    with open("config.json", "r") as f:
        config = json.load(f)
    yo = YoloDetect(model_path=config['model_path'], poly=config['poly'], conf_thresh=config['conf_thresh'])
    cap = cv2.VideoCapture(config['cam_idx'])
    assert(cap.isOpened()==True)
    
    # loop to get frames and predict
    while 1:
        ret, frame = cap.read()
        
        if not ret: 
            break
        
        # perform track
        test(frame, yo, 1)
        
        key = cv2.waitKey(1)
        if key == ord('q'): break
        
    # Release video sources
    cap.release()
    
if __name__ == "__main__":
    main()