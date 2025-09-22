from .BaslerYoloMeasurer import BaslerYoloMeasurer

if __name__ == "__main__":
    measurer = BaslerYoloMeasurer("runs/segment/weld_seg_0911_1-/weights/best.pt")
    while True:
        cmd = input("Type 'go' to grab and measure, 'q' to quit: ")
        if cmd == "q": break
        if cmd == "go":
            dist = measurer.measure()
            print("Distance:", dist, "mm")
    measurer.close()
