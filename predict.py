from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Tải mô hình YOLO
model = YOLO('D:\PY_Code\Test\\runs\detect\\train4\weights\\best.pt')

# Khởi tạo camera (0 là camera mặc định)
cap = cv2.VideoCapture(0)

# Kiểm tra nếu không mở được camera
if not cap.isOpened():
    print("Không thể mở camera")
    exit()

while True:
    # Đọc từng khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình từ camera")
        break

    # Dự đoán trên khung hình
    results = model(frame)

    # Hiển thị kết quả dự đoán
    for r in results:
        # Vẽ kết quả lên khung hình
        frame = r.plot()

    # Hiển thị khung hình với kết quả dự đoán
    cv2.imshow("YOLO Detection", frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
