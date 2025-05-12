import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Đường dẫn đến dữ liệu train và test
train_data_path = "traindata"
test_data_path = "testdata"

# Chuẩn bị dữ liệu train
X_train = []
y_train = []
label_counts = {"fist": 0, "palm": 0, "ok": 0, "point": 0}

for file_name in os.listdir(train_data_path):
    if file_name.endswith(".npy"):
        label_file = file_name.replace(".npy", ".txt")
        features = np.load(os.path.join(train_data_path, file_name))
        with open(os.path.join(train_data_path, label_file), 'r') as f:
            label = f.read().strip()
        X_train.append(features)
        y_train.append(label)
        label_counts[label] += 1

X_train = np.array(X_train)
y_train = np.array(y_train)

# Chuẩn bị dữ liệu test
X_test = []
y_test = []
test_label_counts = {"fist": 0, "palm": 0, "ok": 0, "point": 0}
test_files = {"fist": [], "palm": [], "ok": [], "point": []}

for file_name in os.listdir(test_data_path):
    if file_name.endswith(".npy"):
        label_file = file_name.replace(".npy", ".txt")
        features = np.load(os.path.join(test_data_path, file_name))
        with open(os.path.join(test_data_path, label_file), 'r') as f:
            label = f.read().strip()
        X_test.append(features)
        y_test.append(label)
        test_label_counts[label] += 1
        test_files[label].append(file_name)

X_test = np.array(X_test)
y_test = np.array(y_test)

# In thông tin phân bố nhãn
print("Phân bố dữ liệu train theo nhãn:")
for label, count in label_counts.items():
    print(f"{label}: {count} mẫu")
print(f"Tổng số mẫu train: {len(X_train)}")
print(f"Kích thước đặc trưng: {X_train.shape[1]} chiều")
print("\nPhân bố dữ liệu test theo nhãn:")
for label, count in test_label_counts.items():
    print(f"{label}: {count} mẫu")
print(f"Tổng số mẫu test: {len(X_test)}")

# Đo thời gian huấn luyện
start_time = time.time()

# Huấn luyện SVM trên toàn bộ dữ liệu train
#SVC là viết tắt của Support Vector Classification, và nó là class đại diện cho mô hình SVM phân loại
model = SVC(kernel='linear')
model.fit(X_train, y_train)

end_time = time.time()
training_time = end_time - start_time
print(f"\nThời gian huấn luyện: {training_time:.2f} giây")

# Cross-validation 5-fold để đánh giá trên tập train
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Độ chính xác cross-validation (5-fold) trên tập train: {cv_scores * 100}")
print(f"Trung bình độ chính xác: {cv_scores.mean() * 100:.2f}%")
print(f"Độ lệch chuẩn độ chính xác: {cv_scores.std() * 100:.2f}%")

# Kiểm tra trên tập test
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác tổng thể trên tập test: {test_accuracy * 100:.2f}%")

# Tính tỉ lệ phát hiện đúng cho từng ký tự
detection_rates = {}
total_per_label = {label: len(test_files[label]) for label in test_label_counts}
correct_per_label = {"fist": 0, "palm": 0, "ok": 0, "point": 0}

for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        correct_per_label[y_test[i]] += 1

for label in test_label_counts:
    detection_rate = (correct_per_label[label] / total_per_label[label]) * 100 if total_per_label[label] > 0 else 0
    detection_rates[label] = detection_rate
    print(f"Tỉ lệ phát hiện đúng cho {label}: {detection_rate:.2f}% ({correct_per_label[label]}/{total_per_label[label]})")

average_detection_rate = sum(detection_rates.values()) / len(detection_rates) if detection_rates else 0
print(f"Tỉ lệ trung bình phát hiện đúng: {average_detection_rate:.2f}%")

# Tạo confusion matrix cho tập test
cm = confusion_matrix(y_test, y_pred, labels=["fist", "palm", "ok", "point"])

# Lưu thông số đánh giá vào file
# Cần tạo file này trước khi chạy
with open("exported_model/evaluation_results.txt", "w",  encoding="utf-8" ) as f:
    f.write("=== Kết quả đánh giá mô hình SVM ===\n\n")
    f.write("Phân bố dữ liệu train:\n")
    for label, count in label_counts.items():
        f.write(f"{label}: {count} mẫu\n")
    f.write(f"Tổng số mẫu train: {len(X_train)}\n")
    f.write(f"Kích thước đặc trưng: {X_train.shape[1]} chiều\n\n")
    f.write("Phân bố dữ liệu test:\n")
    for label, count in test_label_counts.items():
        f.write(f"{label}: {count} mẫu\n")
    f.write(f"Tổng số mẫu test: {len(X_test)}\n\n")
    f.write(f"Thời gian huấn luyện: {training_time:.2f} giây\n\n")
    f.write(f"Độ chính xác cross-validation (5-fold) trên tập train: {cv_scores * 100}\n")
    f.write(f"Trung bình độ chính xác: {cv_scores.mean() * 100:.2f}%\n")
    f.write(f"Độ lệch chuẩn độ chính xác: {cv_scores.std() * 100:.2f}%\n\n")
    f.write(f"Độ chính xác tổng thể trên tập test: {test_accuracy * 100:.2f}%\n\n")
    f.write("Tỉ lệ phát hiện đúng cho từng ký tự:\n")
    for label, rate in detection_rates.items():
        f.write(f"{label}: {rate:.2f}% ({correct_per_label[label]}/{total_per_label[label]})\n")
    f.write(f"Tỉ lệ trung bình phát hiện đúng: {average_detection_rate:.2f}%\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm, separator=', '))

# Lưu mô hình
os.makedirs("exported_model", exist_ok=True)
joblib.dump(model, "exported_model/asl_four_svm_model.pkl")
print("Mô hình đã lưu tại: exported_model/asl_four_svm_model.pkl")

# Tạo và lưu biểu đồ phân bố nhãn (train)
plt.figure(figsize=(8, 6))
plt.bar(label_counts.keys(), label_counts.values(), color=['blue', 'green', 'red', 'purple'])
plt.title("Phân bố dữ liệu train theo nhãn cử chỉ")
plt.xlabel("Cử chỉ")
plt.ylabel("Số lượng mẫu")
plt.savefig("exported_model/train_label_distribution.png")
plt.close()

# Tạo và lưu biểu đồ phân bố nhãn (test)
plt.figure(figsize=(8, 6))
plt.bar(test_label_counts.keys(), test_label_counts.values(), color=['blue', 'green', 'red', 'purple'])
plt.title("Phân bố dữ liệu test theo nhãn cử chỉ")
plt.xlabel("Cử chỉ")
plt.ylabel("Số lượng mẫu")
plt.savefig("exported_model/test_label_distribution.png")
plt.close()

# Tạo và lưu biểu đồ độ chính xác cross-validation
plt.figure(figsize=(8, 6))
plt.boxplot(cv_scores * 100, vert=True)
plt.title("Độ chính xác cross-validation (5-fold) trên tập train")
plt.ylabel("Độ chính xác (%)")
plt.savefig("exported_model/cv_accuracy.png")
plt.close()

# Tạo và lưu confusion matrix cho tập test
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["fist", "palm", "ok", "point"], yticklabels=["fist", "palm", "ok", "point"])
plt.title("Confusion Matrix trên tập test")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.savefig("exported_model/confusion_matrix_test.png")
plt.close()