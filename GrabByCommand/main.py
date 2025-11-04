# main
# from .BaslerYoloMeasurer import BaslerYoloMeasurer
from .BaslerYoloM_1 import BaslerYoloMeasurer
import cv2
# import time

if __name__ == "__main__":
    measurer = BaslerYoloMeasurer(
        "runs/segment/weld_seg_1103-2/weights/best.pt",
        draw_masks=False,
        draw_distance=True,  # dots and distance line between them
    )
    while True:

        # if need to display image constantly, can call show_one_frame function like this:
        # image = measurer.show_one_frame()
        # cv2.imshow('Original frames', image)
        # time.sleep(0.2)

        cmd = input("Type 'go' to grab and measure, 'q' to quit: ")
        if cmd == "q": break
        if cmd == "go":
            dist, image = measurer.measure()
            print("Distance:", dist, "mm")
            cv2.imshow('Image test', image)
    measurer.close()
