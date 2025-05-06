# train_and_evaluate.py

import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ==== Configuration ====
train_image_dir = r"C:\Users\TAMIZH MANI\Desktop\Project\cv_project\dataset\guns-knives-yolo\train\images"
train_label_dir = r"C:\Users\TAMIZH MANI\Desktop\Project\cv_project\dataset\guns-knives-yolo\train\labels"
val_image_dir = r"C:\Users\TAMIZH MANI\Desktop\Project\cv_project\dataset\guns-knives-yolo\valid\images"
val_label_dir = r"C:\Users\TAMIZH MANI\Desktop\Project\cv_project\dataset\guns-knives-yolo\valid\labels"
roi_size = (64, 128)
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

# ==== Utility Functions ====

def load_data(image_dir, label_dir, hog_params, roi_size):
    features, labels = [], []
    for img_file in os.listdir(image_dir):
        if not img_file.endswith(('.jpg', '.jpeg', '.png')):
            continue
        image_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, img_file.rsplit('.', 1)[0] + '.txt')
        if not os.path.exists(label_path):
            continue
        image = cv2.imread(image_path)
        if image is None:
            continue
        h_img, w_img, _ = image.shape
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, xc, yc, w, h = map(float, parts[:5])
                if int(cls) not in [0, 1]:
                    continue
                x1 = max(0, int((xc - w / 2) * w_img))
                y1 = max(0, int((yc - h / 2) * h_img))
                x2 = min(w_img, int((xc + w / 2) * w_img))
                y2 = min(h_img, int((yc + h / 2) * h_img))
                roi = image[y1:y2, x1:x2]
                if roi.shape[0] < 20 or roi.shape[1] < 20:
                    continue
                roi_gray = cv2.cvtColor(cv2.resize(roi, roi_size), cv2.COLOR_BGR2GRAY)
                hog_feat = hog(roi_gray, **hog_params)
                features.append(hog_feat)
                labels.append(int(cls))
    return np.array(features), np.array(labels)

# ==== Train + Evaluate ====

print("Loading data...")
X_train, y_train = load_data(train_image_dir, train_label_dir, hog_params, roi_size)
X_val, y_val = load_data(val_image_dir, val_label_dir, hog_params, roi_size)

print("Training SVM...")
svm_clf = SVC(kernel='linear', probability=True)
svm_clf.fit(X_train, y_train)

print("Training Random Forest...")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

def iou(boxA, boxB):
    # box = [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


# ==== Evaluate ====

import json

def evaluate_model(name, clf, X_val, y_val):
    print(f"\n=== {name} Evaluation ===")
    accuracy = clf.score(X_val, y_val)
    y_pred = clf.predict(X_val)

    report = classification_report(y_val, y_pred, target_names=["Gun", "Knife"], output_dict=True)
    conf_matrix = confusion_matrix(y_val, y_pred).tolist()

    print("Validation Accuracy:", accuracy)
    print("Confusion Matrix:\n", np.array(conf_matrix))

    # Placeholder IOU list
    iou_scores = []

    # --- NOTE: True IOU calculation would require actual & predicted bounding boxes ---
    # Here we use dummy IOU = 1 for correct classification, 0 otherwise, just for demonstration
    for true, pred in zip(y_val, y_pred):
        iou_scores.append(1.0 if true == pred else 0.0)
    avg_iou = np.mean(iou_scores)

    metrics = {
        "model": name,
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "avg_iou": avg_iou,
        "classification_report": report
    }

    return metrics
results = {}
results["SVM"] = evaluate_model("SVM", svm_clf, X_val, y_val)
results["RandomForest"] = evaluate_model("Random Forest", rf_clf, X_val, y_val)

# Save results to JSON
with open("evaluation_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("📊 Evaluation metrics saved to evaluation_metrics.json")


# ==== Save ====
joblib.dump(svm_clf, "svm_weapon_model.pkl")
joblib.dump(rf_clf, "rf_weapon_model.pkl")
print("✅ Models saved.")
