import cv2
import numpy as np
import mediapipe as mp
import joblib

# Tải mô hình
model = joblib.load("D:/AI/train3/exported_model/asl_four_svm_model.pkl")

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)

# Khởi tạo webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở webcam!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình!")
        break

    # Chuyển đổi màu sang RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Trích xuất landmarks
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Chuyển landmarks thành mảng 2D (x, y, z)
            features = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten().reshape(1, -1)
            # Dự đoán cử chỉ
            gesture = model.predict(features)[0]
            cv2.putText(frame, f"Cử chỉ: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Thử nghiệm đã kết thúc.")