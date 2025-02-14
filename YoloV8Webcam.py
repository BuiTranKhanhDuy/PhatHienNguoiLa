import cv2
import numpy as np
import os
import time
from ultralytics import YOLO

# Tải mô hình YOLOv8 (pre-trained)
model = YOLO('yolov8n.pt')  # Bạn có thể thay thế 'yolov8n.pt' bằng mô hình khác nếu cần

# Tạo thư mục Data để lưu trữ ảnh
output_directory = "Data"
os.makedirs(output_directory, exist_ok=True)

# Biến để đánh số thứ tự ảnh
image_counter = 0

# Khởi tạo webcam
cap = cv2.VideoCapture(0)  # 0 là ID mặc định của webcam; dùng 1 hoặc số khác nếu có nhiều camera
if not cap.isOpened():
    print("Không thể mở webcam.")
    exit()

try:
    while True:
        ret, img = cap.read()  # Đọc khung hình từ webcam
        if not ret:
            print("Không thể nhận khung hình từ webcam.")
            break

        # Nhận diện đối tượng với YOLOv8
        results = model(img)

        # Duyệt qua các đối tượng phát hiện
        person_detected = False
        for result in results[0].boxes.data:  # Lấy tất cả các đối tượng phát hiện
            if result[5] == 0:  # class 0 là con người trong COCO dataset
                person_detected = True
                x1, y1, x2, y2 = map(int, result[:4])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, '', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Nếu phát hiện con người, lưu ảnh
        # if person_detected:
        #     print("Có người lạ!")
        #     image_path = os.path.join(output_directory, f"detected_person_{image_counter}.jpg")
        #     cv2.imwrite(image_path, img)  # Lưu ảnh
        #     image_counter += 1

        # Hiển thị ảnh từ webcam
        cv2.imshow("Webcam Live Stream with YOLOv8", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
            break

        # Thêm delay để giảm FPS
        time.sleep(0.1)  # 100ms = 10 FPS
except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")
finally:
    # Giải phóng tài nguyên và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()
