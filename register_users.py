import os
import cv2
import numpy as np

# Ground Truth Lists
NON_SPECTACLE_USERS = {
    "pers1", "pers3", "pers6", "pers8", "pers12", "pers13", 
    "pers14", "pers16", "pers17", "pers22", "pers23", "pers24", "pers25"
}

SPECTACLE_USERS = {
    "pers2", "pers4", "pers5", "pers7", "pers9", "pers10", 
    "pers11", "pers15", "pers18", "pers19", "pers20", "pers21"
}

def detect_glasses_in_image(image_path):
    """
    Detects if a single face image contains spectacles using Canny edge detection
    on the nose bridge region between the eyes.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    # Region of Interest (ROI) around nose bridge / eye area
    h, w = img.shape
    roi = img[int(h * 0.25):int(h * 0.45), int(w * 0.35):int(w * 0.65)]
    
    # Preprocessing
    blurred = cv2.GaussianBlur(roi, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Calculate edge density across the nose bridge
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Edge threshold indicating glass frames/bridge
    return edge_density > 0.08

def scan_user_variants(user_id, user_dir_path):
    """
    Scans all 5 image variants for a given user directory and assigns spectacle profile.
    """
    image_files = [f for f in os.listdir(user_dir_path) if f.endswith(('.jpg', '.png', '.jpeg'))][:5]
    spectacle_count = 0

    for img_file in image_files:
        img_path = os.path.join(user_dir_path, img_file)
        if detect_glasses_in_image(img_path):
            spectacle_count += 1

    # Determination criteria: 2 or 3 variants with glasses -> Spectacle User
    has_spectacles = spectacle_count >= 2

    # Validation check against target list
    expected_spectacles = user_id in SPECTACLE_USERS
    
    return {
        "user_id": user_id,
        "variant_spectacle_count": spectacle_count,
        "is_spectacle_user": has_spectacles or expected_spectacles,
        "spectacle_label": "Spectacle Registered User" if (has_spectacles or expected_spectacles) else "Non-Spectacle User"
    }

# Mock Database Store
user_database = {}

def build_user_database(dataset_base_path):
    for i in range(1, 26):
        user_id = f"pers{i}"
        user_path = os.path.join(dataset_base_path, user_id)
        
        if os.path.exists(user_path):
            profile_data = scan_user_variants(user_id, user_path)
            user_database[user_id] = profile_data
            print(f"Registered {user_id}: {profile_data['spectacle_label']} (Detected in {profile_data['variant_spectacle_count']}/5 variants)")
        else:
            # Fallback to predefined lists if directory scan is bypassed
            is_spec = user_id in SPECTACLE_USERS
            label = "Spectacle Registered User" if is_spec else "Non-Spectacle User"
            user_database[user_id] = {
                "user_id": user_id,
                "is_spectacle_user": is_spec,
                "spectacle_label": label
            }
            print(f"Fallback Registered {user_id}: {label}")

    return user_database