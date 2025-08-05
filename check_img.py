from ultralytics import YOLO
import cv2
onnx_model = YOLO("best.pt")
test_data = "test_media/cardboard.jpg"

if __name__ == "__main__":
    print("success")
    result = onnx_model(test_data)
    cv2.namedWindow("DISPLAY",cv2.WINDOW_AUTOSIZE)
    while True:
        cv2.imshow("DISPLAY",result[0].plot())
        keys = cv2.waitKey(1)
        if keys == 27:
            cv2.destroyAllWindows()
            break