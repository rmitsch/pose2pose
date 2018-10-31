"""
Example script using PyOpenPose.
"""
import argparse
import PyOpenPose as OP
import time
import cv2
import os

OPENPOSE_ROOT = os.environ["OPENPOSE_ROOT"]

def run():

    cap = cv2.VideoCapture(args.filename)
    with_face = with_hands = True
    op = OP.OpenPose((656, 368), (368, 368), (1280, 720), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0,
                      False, OP.OpenPose.ScaleMode.ZeroToOne, with_face, with_hands)
    paused = False
    delay = {True: 0, False: 1}

    count = 0
    try:
        ret, rgb = cap.read()
        
        while ret:
            t = time.time()
            op.detectPose(rgb)
            op.detectFace(rgb)
            op.detectHands(rgb)
            t = time.time() - t

            res = op.render(rgb)
            persons = op.getKeypoints(op.KeypointType.POSE)[0]

            if persons is not None and len(persons) == 1:
                gray = cv2.cvtColor(res-rgb, cv2.COLOR_RGB2GRAY)
                ret_threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

                count += 1
                print(count)

                cv2.imwrite("original/{}.png".format(count), rgb)
                cv2.imwrite("landmarks/{}.png".format(count), binary)
            
            ret, rgb = cap.read()
            
    except Exception as e:
        print(e)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', dest='filename', type=str, help='Name of the video file.')
    args = parser.parse_args()
    if not os.path.exists(os.path.join('./', 'original')):
        os.makedirs(os.path.join('./', 'original'))
    if not os.path.exists(os.path.join('./', 'landmarks')):
        os.makedirs(os.path.join('./', 'landmarks'))
    
    run()
