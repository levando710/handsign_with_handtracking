import cv2
import os

# === Thông tin đầu vào ===
video_path = r"video.mp4"        # đường dẫn đến video bạn muốn cắt
output_folder = r"imgtestdata"       # thư mục lưu ảnh
frame_interval = 5                    # lưu mỗi 5 frame (giảm số ảnh nếu video quá dài)

# === Tạo thư mục lưu ảnh nếu chưa có ===
os.makedirs(output_folder, exist_ok=True)

# === Mở video ===
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không thể mở video!")
    exit()

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"✅ Đã lưu {saved_count} ảnh vào thư mục '{output_folder}'")
