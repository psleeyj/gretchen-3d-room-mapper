import cv2

print("Searching for cameras...\n")

found = False

for index in range(10):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        cap.release()
        continue

    ret, frame = cap.read()

    if ret:
        found = True
        print(f"[FOUND] Camera index {index}")
        print("Press any key in the preview window to continue.\n")

        cv2.imshow(f"Camera {index}", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cap.release()

if not found:
    print("No working cameras found.")
