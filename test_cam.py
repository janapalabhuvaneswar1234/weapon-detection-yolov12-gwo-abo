import cv2

for i in range(5):
    print(f"Testing index {i}...")

    # Use safer backend
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"❌ Index {i} not working")
        continue

    print(f"✅ Camera found at index {i}")

    ret, frame = cap.read()
    if ret:
        cv2.imshow(f"Camera {i}", frame)
        cv2.waitKey(2000)  # show for 2 seconds
        cv2.destroyAllWindows()
    else:
        print("⚠ Could not read frame")

    cap.release()
    break