import os
import cv2
import numpy as np
from skimage.feature import hog
import joblib

# ==== Load the Trained SVM Model ====
clf = joblib.load("svm_weapon_model.pkl")

# ==== Parameters ====
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}
roi_size = (64, 128)
step_size = 32  # updated here as well
nms_thresh = 0.3
conf_thresh = 0.8

# ==== Non-Maximum Suppression ====
def non_max_suppression_fast(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 5]

    idxs = np.argsort(scores)[::-1]

    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        overlap = (w * h) / ((x2[i] - x1[i]) * (y2[i] - y1[i]) + 1e-6)
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlapThresh)[0] + 1)))

    return boxes[pick].astype("int")

# ==== Sliding Window Detection ====
def sliding_window_detection(image, clf, hog_params, roi_size, conf_thresh=0.8, step_size=32):
    detected_boxes = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for y in range(0, gray.shape[0] - roi_size[1], step_size):
        for x in range(0, gray.shape[1] - roi_size[0], step_size):
            window = gray[y:y+roi_size[1], x:x+roi_size[0]]
            if window.shape[0] != roi_size[1] or window.shape[1] != roi_size[0]:
                continue
            features = hog(window, **hog_params).reshape(1, -1)
            pred_class = clf.predict(features)[0]
            if hasattr(clf, "predict_proba"):
                confidence = clf.predict_proba(features).max()
            else:
                confidence = 1.0
            if confidence > conf_thresh:
                x2 = x + roi_size[0]
                y2 = y + roi_size[1]
                detected_boxes.append([x, y, x2, y2, pred_class, confidence])
    return detected_boxes


# ==== Weapon Detection Function ====
def detect_weapons_in_image(image_path):
    print("[INFO] Loading image from:", image_path)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image could not be loaded.")

    
    print("[INFO] Image received.")
    orig_image = image.copy()

    detected_boxes = sliding_window_detection(
        image, clf, hog_params, roi_size,
        conf_thresh=conf_thresh, step_size=step_size
    )

    # Apply NMS
    nms_boxes = non_max_suppression_fast(np.array(detected_boxes), overlapThresh=nms_thresh) if detected_boxes else []

    # Draw boxes
    # Draw only the most confident box
    if len(nms_boxes) > 0:
        # Sort by confidence (descending)
        nms_boxes = sorted(nms_boxes, key=lambda x: x[5], reverse=True)
        x1, y1, x2, y2, cls, conf = nms_boxes[0]
        label = "Gun" if cls == 0 else "Knife"
        color = (0, 255, 0) if cls == 0 else (0, 0, 255)
        cv2.rectangle(orig_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(orig_image, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # Save result
    os.makedirs("outputs", exist_ok=True)
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join("outputs", f"{name}_output{ext}")
    cv2.imwrite(output_path, orig_image)

    print(f"[INFO] Detection completed. Result saved to: {output_path}")
    return output_path
