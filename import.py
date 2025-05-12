import os
import cv2
import mediapipe as mp
import numpy as np

# Kiểm tra cài đặt MediaPipe
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
    print("MediaPipe Hands đã được khởi tạo thành công.")
except Exception as e:
    print(f"LỖI: Không thể khởi tạo MediaPipe Hands: {e}")
    exit()

# Đường dẫn đến dataset và output
dataset_path = r"imgtestdata"
output_path = "testdata"

# Kiểm tra thư mục dataset
if not os.path.exists(dataset_path):
    print(f"LỖI: Thư mục dataset không tồn tại tại: {dataset_path}")
    print("Vui lòng kiểm tra đường dẫn hoặc tải dataset về đúng vị trí.")
    exit()

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(output_path, exist_ok=True)
print(f"Thư mục output đã được tạo tại: {output_path}")

# Ánh xạ các thư mục cử chỉ thành nhãn (dựa trên giả định của bạn)
gesture_mapping = {
    "A": "fist",
    "B": "palm",
    "C": "ok",
    "D": "point"  # Giả định D là point, bạn có thể thay đổi nếu khác
}

# Kiểm tra các thư mục trong dataset
folders_found = os.listdir(dataset_path)
print(f"Các thư mục tìm thấy trong {dataset_path}: {folders_found}")

# Xử lý từng ảnh
files_processed = 0
files_skipped = 0
for folder in folders_found:
    if folder in gesture_mapping:
        gesture_label = gesture_mapping[folder]
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            print(f"THÔNG BÁO: {folder_path} không phải là thư mục, bỏ qua.")
            continue

        # Kiểm tra các file trong thư mục
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Thư mục {folder} ({gesture_label}): Tìm thấy {len(image_files)} ảnh.")

        if len(image_files) == 0:
            print(f"LỖI: Không có ảnh nào trong thư mục {folder}!")
            continue

        for file_name in image_files:
            file_path = os.path.join(folder_path, file_name)
            try:
                # Đọc ảnh
                img = cv2.imread(file_path)
                if img is None:
                    print(f"LỖI: Không thể đọc ảnh {file_name}")
                    files_skipped += 1
                    continue
                print(f"Đã đọc thành công ảnh: {file_name}, kích thước: {img.shape}")

                # Chuyển đổi màu sang RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Trích xuất landmarks
                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Chuyển landmarks thành mảng 2D (x, y, z)
                        features = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                        # Tạo tên file đầu ra
                        output_npy = os.path.join(output_path, f"{gesture_label}_{file_name.replace('.', '_').replace(' ', '_')}.npy")
                        output_txt = os.path.join(output_path, f"{gesture_label}_{file_name.replace('.', '_').replace(' ', '_')}.txt")
                        # Lưu file .npy
                        np.save(output_npy, features)
                        # Lưu file .txt với nhãn
                        with open(output_txt, 'w') as f:
                            f.write(gesture_label)
                        files_processed += 1
                        print(f"Đã xử lý: {file_name} -> {gesture_label}")
                else:
                    print(f"THÔNG BÁO: Không phát hiện bàn tay trong {file_name}")
                    files_skipped += 1
            except Exception as e:
                print(f"LỖI khi xử lý {file_name}: {e}")
                files_skipped += 1

if files_processed == 0:
    print("LỖI: Không có file nào được xử lý. Vui lòng kiểm tra dataset hoặc ánh sáng trong ảnh.")
else:
    print(f"Xử lý hoàn tất. Đã xử lý {files_processed} file, bỏ qua {files_skipped} file. Kết quả được lưu tại: {output_path}")