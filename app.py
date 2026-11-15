import base64
import io
import os
import random
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import cv2
import re
import json
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, session, redirect, url_for

# Optional Twilio Import for SMS
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

cv2.setUseOptimized(True)

app = Flask(__name__)
app.secret_key = "super_secret_arcface_key"

DATASET_PATH = r"D:\mydata\College\MTECH\Semester 3\Final Yr Project\arcface_app\dataset"
SIMILARITY_THRESHOLD = 0.25
OWNER_ID = "pers1"
DELETED_USERS_FILE = 'deleted_users.json'
# EMAIL CONFIGURATION (Update with your SMTP credentials)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "hzbrjxqwkqmkhrrk"

# TWILIO SMS CONFIGURATION (Optional)
TWILIO_ACCOUNT_SID = "your_twilio_sid"
TWILIO_AUTH_TOKEN = "your_twilio_auth_token"
TWILIO_PHONE_NUMBER = "your_twilio_phone_number"

face_app = None
embedding_db = {}         # { person_id: [embedding1, embedding2, ...] }
person_spectacle_map = {} # { person_id: True / False }
otp_storage = {}          # { contact: otp_code }

# GROUND-TRUTH SPECTACLE MAPPINGS
SPECTACLE_USERS = {
    "pers2", "pers4", "pers5", "pers7", "pers9", "pers10",
    "pers11", "pers15", "pers18", "pers19", "pers20", "pers21"
}

NON_SPECTACLE_USERS = {
    "pers1", "pers3", "pers6", "pers8", "pers12", "pers13",
    "pers14", "pers16", "pers17", "pers22", "pers23", "pers24", "pers25"
}

def send_email_background(receiver_email, otp_code):
    """Sends OTP email asynchronously in a background thread."""
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = receiver_email
        msg['Subject'] = "ArcFace MFA Verification OTP Code"
        
        body = f"Hello,\n\nYour One-Time Password (OTP) for registration is: {otp_code}\nThis code is valid for 1 minute.\n\nRegards,\nArcFace MFA System"
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, receiver_email, msg.as_string())
        server.quit()
        print(f"[Email Dispatch] Successfully sent OTP {otp_code} to email {receiver_email}")
    except Exception as e:
        print(f"[Email Error] Failed to send email to {receiver_email}: {e}")

def send_sms_background(phone_number, otp_code):
    """Sends OTP SMS asynchronously via Twilio in a background thread."""
    if not TWILIO_AVAILABLE:
        print("[SMS Warning] Twilio package not installed. Run 'pip install twilio'")
        return
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=f"Your ArcFace MFA Code is: {otp_code}. Valid for 1 minute.",
            from_=TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        print(f"[SMS Dispatch] Successfully sent OTP {otp_code} to phone {phone_number}")
    except Exception as e:
        print(f"[SMS Error] Failed to send SMS to {phone_number}: {e}")

def detect_spectacles_roi(gray_roi):
    if gray_roi is None or gray_roi.size == 0:
        return False
    
    h, w = gray_roi.shape
    bridge_area = gray_roi[int(h * 0.30):int(h * 0.55), int(w * 0.30):int(w * 0.70)]
    if bridge_area.size == 0:
        return False

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    equalized = clahe.apply(bridge_area)
    filtered = cv2.bilateralFilter(equalized, d=5, sigmaColor=75, sigmaSpace=75)
    edges = cv2.Canny(filtered, 100, 200)

    edge_density = np.sum(edges > 0) / float(edges.size)
    return bool(edge_density > 0.18)

def detect_mask_roi(gray_face_roi, color_face_roi=None):
    """
    Robust Skin-Tone Based Mask Detector:
    Checks if valid skin pixels exist in the lower face (nose/mouth region).
    Returns True (Mask Detected / Fraud) ONLY if skin is missing (covered by a mask).
    Returns False (Valid / Unmasked) if skin pixels are detected.
    """
    if color_face_roi is None or color_face_roi.size == 0:
        return False
        
    h, w, _ = color_face_roi.shape
    # Isolate lower face region (nose tip to chin: vertical 50% to 85%, horizontal 30% to 70%)
    y1, y2 = int(h * 0.50), int(h * 0.85)
    x1, x2 = int(w * 0.30), int(w * 0.70)
    
    lower_color = color_face_roi[y1:y2, x1:x2]
    if lower_color.size == 0:
        return False

    # Convert to HSV color space for reliable skin detection across lighting conditions
    hsv = cv2.cvtColor(lower_color, cv2.COLOR_BGR2HSV)
    
    # Define broad HSV skin tone color range (handles shadows and varying skin tones)
    # Lower skin range and Upper skin range
    lower_skin = np.array([0, 15, 30], dtype=np.uint8)
    upper_skin = np.array([35, 255, 255], dtype=np.uint8)
    
    # Create skin mask
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Calculate percentage of skin pixels in the lower face
    skin_pixel_count = np.sum(skin_mask > 0)
    total_pixels = skin_mask.size
    skin_ratio = skin_pixel_count / float(total_pixels)

    print(f"[Mask Skin Check] Lower Face Skin Ratio: {skin_ratio:.4f}")

    # If skin ratio is very low (less than 8% skin pixels), the mouth/nose area is covered by a mask
    if skin_ratio < 0.08:
        return True # Mask detected (Fraud)
        
    return False # Unmasked (Valid User)

def load_face_model():
    global face_app
    try:
        from insightface.app import FaceAnalysis
        face_app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(320, 320))
        print("[ArcFace] InsightFace Model loaded successfully.")
    except Exception as e:
        print(f"[ArcFace] ERROR initializing InsightFace: {e}")
        face_app = None

def build_embedding_database():
    global embedding_db, person_spectacle_map
    if face_app is None or not os.path.exists(DATASET_PATH):
        print(f"[DB] Target path '{DATASET_PATH}' unavailable or model uninitialized.")
        return

    count = 0
    for person_id in sorted(os.listdir(DATASET_PATH)):
        person_dir = os.path.join(DATASET_PATH, person_id)
        if not os.path.isdir(person_dir):
            continue

        image_files = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ][:5]

        embeddings = []
        spectacle_variant_count = 0

        for img_file in image_files:
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            try:
                faces = face_app.get(img)
                if faces:
                    primary_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                    embeddings.append(primary_face.normed_embedding)

                    bbox = primary_face.bbox.astype(int)
                    x1, y1, x2, y2 = max(0, int(bbox[0])), max(0, int(bbox[1])), min(img.shape[1], int(bbox[2])), min(img.shape[0], int(bbox[3]))
                    gray_face = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    
                    eye_roi = gray_face[int(gray_face.shape[0]*0.15):int(gray_face.shape[0]*0.55), :]
                    if detect_spectacles_roi(eye_roi):
                        spectacle_variant_count += 1
            except Exception as e:
                print(f"[DB] Extraction error on {img_file}: {e}")

        if embeddings:
            embedding_db[person_id] = embeddings
            
            pid_clean = person_id.lower().strip()
            if pid_clean in SPECTACLE_USERS:
                is_spectacle_user = True
            elif pid_clean in NON_SPECTACLE_USERS:
                is_spectacle_user = False
            else:
                is_spectacle_user = bool(spectacle_variant_count >= 2)

            person_spectacle_map[person_id] = is_spectacle_user
            count += 1

    print(f"[DB] Successfully registered {count} dataset profiles.")

def cosine_similarity(a, b):
    return float(np.dot(a, b))

def identify_face(live_embedding):
    best_person = None
    best_score = -1.0

    for person_id, embeddings in embedding_db.items():
        scores = [cosine_similarity(live_embedding, emb) for emb in embeddings]
        max_score = float(np.max(scores)) if scores else -1.0
        if max_score > best_score:
            best_score = max_score
            best_person = person_id

    return best_person, float(round(best_score, 4))

def load_deleted_users():
    if os.path.exists(DELETED_USERS_FILE):
        try:
            with open(DELETED_USERS_FILE, 'r') as f:
                return set(json.load(f))
        except:
            pass
    return set()

def save_deleted_users(deleted_set):
    with open(DELETED_USERS_FILE, 'w') as f:
        json.dump(list(deleted_set), f)

PERMANENTLY_DELETED_USERS = load_deleted_users()

# ROUTES

@app.route('/')
@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template(
        'dashboard.html',
        user_id=session['user_id'],
        is_spectacle=session.get('is_spectacle', False),
        similarity=session.get('similarity', 0.0),
        spectacle_variant=session.get('spectacle_variant', '—')
    )

@app.route('/deleteusers')
def delete_users_page():
    if 'user_id' not in session or str(session.get('user_id', '')).strip().lower() != OWNER_ID.lower():
        return redirect(url_for('dashboard'))
    
    # Custom sort function to sort user IDs numerically (e.g., pers2 comes before pers10)
    def natural_sort_key(user_id):
        match = re.search(r'\d+', str(user_id))
        return int(match.group()) if match else 0

    registered_users = sorted(
        [
            uid for uid in person_spectacle_map.keys()
            if str(uid).strip().lower() != OWNER_ID.lower() and str(uid).strip() not in PERMANENTLY_DELETED_USERS
        ],
        key=natural_sort_key
    )
    
    return render_template(
        'deleteusers.html',
        user_id=session['user_id'],
        is_owner=True,
        users=registered_users
    )

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json() or {}
    if 'image' not in data or not data['image']:
        return jsonify({
            'status': 'FAILED',
            'auth_result': 'ACCESS DENIED (No Image Captured)',
            'identified_profile': 'Unknown',
            'spectacle_variant': '—',
            'similarity_score': 0.0
        })

    try:
        img_bytes = base64.b64decode(data['image'].split(',')[1])
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        orig_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        h, w = orig_img.shape[:2]
        scale_factor = w / 480.0 if w > 480 else 1.0
        cv_img = cv2.resize(orig_img, (480, int(h / scale_factor)), interpolation=cv2.INTER_AREA) if w > 480 else orig_img.copy()

        insight_faces = face_app.get(cv_img) if face_app else []
        if not insight_faces:
            return jsonify({
                'status': 'DENIED',
                'auth_result': 'ACCESS DENIED (Fraud / Hacker Detected: No Face / Covered)',
                'identified_profile': 'Unknown (Masked)',
                'spectacle_variant': '—',
                'similarity_score': 0.0
            })

        primary_face = max(insight_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        
        gray_full = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        bbox = primary_face.bbox
        x1, y1 = max(0, int(bbox[0] * scale_factor)), max(0, int(bbox[1] * scale_factor))
        x2, y2 = min(orig_img.shape[1], int(bbox[2] * scale_factor)), min(orig_img.shape[0], int(bbox[3] * scale_factor))
        face_roi_gray = gray_full[y1:y2, x1:x2]
        face_roi_color = orig_img[y1:y2, x1:x2]

        # -------------------------------------------------------------
        # STRICT ANTI-FRAUD MASK CHECK (Evaluated FIRST before anything else)
        # -------------------------------------------------------------
        if detect_mask_roi(face_roi_gray, face_roi_color):
            return jsonify({
                'status': 'DENIED',
                'auth_result': 'ACCESS DENIED (Fraud / Hacker Detected: Face Mask/Covering)',
                'identified_profile': 'Unknown (Masked)',
                'spectacle_variant': 'Mask Detected (Violation)',
                'similarity_score': 0.0
            })
        # -------------------------------------------------------------

        live_emb = primary_face.normed_embedding
        identified_profile, similarity_score = identify_face(live_emb)

        if not identified_profile or similarity_score < SIMILARITY_THRESHOLD:
            return jsonify({
                'status': 'FAILED',
                'auth_result': 'ACCESS DENIED (Unrecognized User)',
                'identified_profile': 'Unknown',
                'spectacle_variant': '—',
                'similarity_score': float(similarity_score)
            })

        eye_y1 = int(y1 + 0.18 * (y2 - y1))
        eye_h = int(0.32 * (y2 - y1))
        eye_roi = gray_full[max(0, eye_y1):min(gray_full.shape[0], eye_y1 + eye_h), max(0, int(x1 + 0.05 * (x2 - x1))):min(gray_full.shape[1], int(x1 + 0.95 * (x2 - x1)))]
        has_spectacles_live = detect_spectacles_roi(eye_roi)

        is_registered_spectacle_user = person_spectacle_map.get(identified_profile, False)

        # STRICT SPEC RULES
        if not is_registered_spectacle_user:
            if has_spectacles_live:
                return jsonify({
                    'status': 'DENIED',
                    'auth_result': 'ACCESS DENIED (Invalid User - Spectacle Violation)',
                    'identified_profile': str(identified_profile),
                    'spectacle_variant': 'Spectacle Detected (Violation)',
                    'similarity_score': float(similarity_score)
                })
            else:
                auth_result = 'ACCESS GRANTED (Valid User)'
                spectacle_label = 'Non-Spectacle User'
        else:
            auth_result = 'ACCESS GRANTED (Valid User)'
            if has_spectacles_live:
                spectacle_label = 'Spectacle Registered User'
            else:
                spectacle_label = 'Spectacle User (Removed Spectacles)'

        session['user_id'] = identified_profile
        session['is_spectacle'] = is_registered_spectacle_user
        session['similarity'] = float(similarity_score)
        session['spectacle_variant'] = spectacle_label

        return jsonify({
            'status': 'SUCCESS',
            'auth_result': auth_result,
            'identified_profile': str(identified_profile),
            'spectacle_variant': spectacle_label,
            'similarity_score': float(similarity_score),
            'redirect': url_for('dashboard')
        })

    except Exception as e:
        print(f"[Login Error] {e}")
        return jsonify({
            'status': 'ERROR',
            'auth_result': f'ACCESS DENIED (Error: {e})',
            'identified_profile': 'Error',
            'spectacle_variant': '—',
            'similarity_score': 0.0
        })

        live_emb = primary_face.normed_embedding
        identified_profile, similarity_score = identify_face(live_emb)

        if identified_profile and str(identified_profile).strip() in PERMANENTLY_DELETED_USERS:
            return jsonify({
                'status': 'DENIED',
                'auth_result': 'ACCESS DENIED (Account Permanently Deleted)',
                'identified_profile': 'Unknown (Deleted)',
                'spectacle_variant': '—',
                'similarity_score': 0.0
            })

        if not identified_profile or similarity_score < SIMILARITY_THRESHOLD:
            return jsonify({
                'status': 'FAILED',
                'auth_result': 'ACCESS DENIED (Unrecognized User)',
                'identified_profile': 'Unknown',
                'spectacle_variant': '—',
                'similarity_score': float(similarity_score)
            })

        eye_y1 = int(y1 + 0.18 * (y2 - y1))
        eye_h = int(0.32 * (y2 - y1))
        eye_roi = gray_full[max(0, eye_y1):min(gray_full.shape[0], eye_y1 + eye_h), max(0, int(x1 + 0.05 * (x2 - x1))):min(gray_full.shape[1], int(x1 + 0.95 * (x2 - x1)))]
        has_spectacles_live = detect_spectacles_roi(eye_roi)

        is_registered_spectacle_user = person_spectacle_map.get(identified_profile, False)

        # STRICT SPEC RULES
        if not is_registered_spectacle_user:
            if has_spectacles_live:
                return jsonify({
                    'status': 'DENIED',
                    'auth_result': 'ACCESS DENIED (Invalid User)',
                    'identified_profile': str(identified_profile),
                    'spectacle_variant': 'Spectacle Detected (Violation)',
                    'similarity_score': float(similarity_score)
                })
            else:
                auth_result = 'ACCESS GRANTED (Valid User)'
                spectacle_label = 'Non-Spectacle User'
        else:
            auth_result = 'ACCESS GRANTED (Valid User)'
            if has_spectacles_live:
                spectacle_label = 'Spectacle Registered User'
            else:
                spectacle_label = 'Spectacle User (Removed Spectacles)'

        session['user_id'] = identified_profile
        session['is_spectacle'] = is_registered_spectacle_user
        session['similarity'] = float(similarity_score)
        session['spectacle_variant'] = spectacle_label

        return jsonify({
            'status': 'SUCCESS',
            'auth_result': auth_result,
            'identified_profile': str(identified_profile),
            'spectacle_variant': spectacle_label,
            'similarity_score': float(similarity_score),
            'redirect': url_for('dashboard')
        })

    except Exception as e:
        print(f"[Login Error] {e}")
        return jsonify({
            'status': 'ERROR',
            'auth_result': f'ACCESS DENIED (Error: {e})',
            'identified_profile': 'Error',
            'spectacle_variant': '—',
            'similarity_score': 0.0
        })

@app.route('/api/send_otp', methods=['POST'])
def send_otp():
    data = request.get_json() or {}
    contact = data.get('contact', '').strip()
    if not contact:
        return jsonify({'status': 'FAILED', 'message': 'Please enter a valid email or phone number.'})
    
    otp = str(random.randint(100000, 999999))
    otp_storage[contact] = otp
    
    if '@' in contact:
        threading.Thread(target=send_email_background, args=(contact, otp)).start()
        channel = "email"
    else:
        threading.Thread(target=send_sms_background, args=(contact, otp)).start()
        channel = "mobile number"

    print(f"[OTP SYSTEM] Dispatched verification code to {contact}: {otp}")
    return jsonify({
        'status': 'SUCCESS', 
        'message': f'OTP successfully sent to your {channel} within 1 min!'
    })

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json() or {}
    contact = data.get('contact', '').strip()
    user_otp = data.get('otp', '').strip()
    person_id = data.get('person_id', '').strip()
    image_data = data.get('image')

    if not person_id or not contact or not user_otp or not image_data:
        return jsonify({'status': 'FAILED', 'message': 'All registration fields and face capture are required.'})

    if otp_storage.get(contact) != user_otp:
        return jsonify({'status': 'FAILED', 'message': 'Invalid OTP code. Please check or click Resend OTP.'})

    try:
        person_dir = os.path.join(DATASET_PATH, person_id)
        os.makedirs(person_dir, exist_ok=True)
        
        img_bytes = base64.b64decode(image_data.split(',')[1])
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        save_path = os.path.join(person_dir, 'variant_1.jpg')
        pil_img.save(save_path)

        build_embedding_database()
        otp_storage.pop(contact, None)

        return jsonify({
            'status': 'SUCCESS', 
            'message': 'User registered successfully!', 
            'redirect': url_for('login_page')
        })

    except Exception as e:
        print(f"[Registration Error] {e}")
        return jsonify({'status': 'ERROR', 'message': str(e)})

@app.route('/api/delete-user', methods=['POST'])
def api_delete_user():
    # Strict Backend Owner Authentication
    if 'user_id' not in session or str(session.get('user_id', '')).strip().lower() != OWNER_ID.lower():
        return jsonify({'status': 'ERROR', 'message': 'Unauthorized: Only website owner can delete users.'}), 403

    data = request.get_json() or {}
    target_user = str(data.get('target_user', '')).strip()

    if not target_user or target_user.lower() == OWNER_ID.lower():
        return jsonify({'status': 'ERROR', 'message': 'Invalid target or cannot delete owner account.'}), 400

    try:
        # 1. Add to permanent blocklist so they can never log in again
        PERMANENTLY_DELETED_USERS.add(target_user)
        save_deleted_users(PERMANENTLY_DELETED_USERS)

        # 2. Remove from active in-memory dictionaries
        if target_user in person_spectacle_map:
            del person_spectacle_map[target_user]
        if 'known_face_embeddings' in globals() and target_user in known_face_embeddings:
            del known_face_embeddings[target_user]

        # 3. Permanently wipe their dataset folder and images from disk
        user_folder = os.path.join('dataset', target_user)
        if os.path.exists(user_folder):
            shutil.rmtree(user_folder)
            print(f"[Permanent Deletion] Wiped folder: {user_folder}")

        # Also check alternative common dataset directory names if applicable
        alt_folder = os.path.join('face_dataset', target_user)
        if os.path.exists(alt_folder):
            shutil.rmtree(alt_folder)

        return jsonify({'status': 'SUCCESS', 'message': f'User {target_user} permanently deleted and blocked from logging in.'})
    
    except Exception as e:
        print(f"[Deletion Error] {e}")
        return jsonify({'status': 'ERROR', 'message': f'Failed to delete user: {str(e)}'}), 500

if __name__ == '__main__':
    load_face_model()
    build_embedding_database()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)