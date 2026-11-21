import os
import base64
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO
from manga_ocr import MangaOcr
import google.generativeai as genai
import textwrap
import concurrent.futures # HIZLANDIRICI KÜTÜPHANE

app = Flask(__name__)
CORS(app)

# --- SENİN API KEY'İN ---
GEMINI_API_KEY = "AIzaSyBbWomv4GR3ZkIMsW7uTqZkGxK5jJ2Pm5I"
genai.configure(api_key=GEMINI_API_KEY)

# --- AYARLAR ---
MODEL_PATH = 'comic-speech-bubble-detector.pt' 
ACTIVE_MODEL_NAME = "models/gemini-2.5-flash"

print(f"--- SISTEM BASLATILIYOR ({ACTIVE_MODEL_NAME} - TURBO MODE) ---")

# 1. MODEL
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    print(f"✅ YOLO Hazır")
else:
    model = YOLO('yolov8n.pt')
    print("⚠️ Standart YOLO")

# 2. OCR
print("⏳ OCR Yükleniyor...")
mocr = MangaOcr()
print("✅ OCR Hazır")

def base64_to_cv2(b64str):
    if "," in b64str: b64str = b64str.split(",")[1]
    img_data = base64.b64decode(b64str)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def translate_batch(text_list):
    if not text_list: return []
    joined_text = "\n---NEXT---\n".join(text_list)
    
    safety = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    try:
        model_gemini = genai.GenerativeModel(ACTIVE_MODEL_NAME, safety_settings=safety)
        prompt = (
            "You are a manga translator. Translate the following Japanese lines to English. "
            "Maintain the order strictly. Do not add notes. "
            "Separator is '---NEXT---'. Return translations separated by '---NEXT---'.\n\n"
            f"{joined_text}"
        )
        response = model_gemini.generate_content(prompt)
        translations = response.text.strip().split("---NEXT---")
        translations = [t.strip().replace('\n', ' ') for t in translations]
        
        while len(translations) < len(text_list):
            translations.append("...")
        return translations
    except Exception as e:
        print(f"⚠️ API HATASI: {e}")
        return ["Error"] * len(text_list)

def draw_text_opencv(img, text, x1, y1, x2, y2):
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    bubble_width = x2 - x1
    chars_per_line = max(5, int(bubble_width / 9)) 
    lines = textwrap.wrap(text, width=chars_per_line)
    
    line_height = 20
    total_height = len(lines) * line_height
    current_y = y1 + (y2 - y1 - total_height) // 2 + 15
    
    for line in lines:
        text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
        text_w = text_size[0]
        text_x = x1 + (bubble_width - text_w) // 2
        text_x = max(x1 + 2, text_x)
        cv2.putText(img, line, (text_x, current_y), font, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)
        current_y += line_height
    return img

# --- HIZLANDIRILMIŞ OCR FONKSİYONU ---
def process_single_crop(args):
    """Bu fonksiyon tek bir parçayı alır ve okur (Paralel çalışacak)"""
    crop, index = args
    try:
        text = mocr(crop)
        if not text: text = "..."
        return index, text
    except:
        return index, "..."

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    img_cv = base64_to_cv2(data['image'])
    
    # 1. TESPİT
    results = model(img_cv)
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            if box.conf[0] < 0.25: continue
            detections.append((int(x1), int(y1), int(x2), int(y2)))
    detections.sort(key=lambda x: x[1])

    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    print(f"--- SAYFA: {len(detections)} Balon ---")

    # 2. PARALEL OCR (HIZLANDIRMA BURADA)
    # Tüm kırpılmış resimleri hazırla
    crop_tasks = []
    for i, (x1, y1, x2, y2) in enumerate(detections):
        pad = 5
        crop = img_pil.crop((max(0, x1-pad), max(0, y1-pad), min(img_pil.width, x2+pad), min(img_pil.height, y2+pad)))
        crop_tasks.append((crop, i))

    raw_texts = [""] * len(detections)
    
    # Bilgisayarın çekirdeklerini kullanarak aynı anda okut
    # max_workers=4 idealdır, fazlası CPU'yu boğabilir.
    print("⚡ OCR Paralel Çalışıyor...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_single_crop, crop_tasks))
    
    # Sonuçları doğru sıraya koy
    for index, text in results:
        raw_texts[index] = text

    # 3. TOPTAN ÇEVİRİ
    print("🚀 Google'a Gönderiliyor...")
    translated_texts = translate_batch(raw_texts)
    
    # 4. ÇİZİM
    for i, (x1, y1, x2, y2) in enumerate(detections):
        en_text = translated_texts[i] if i < len(translated_texts) else "..."
        img_cv = draw_text_opencv(img_cv, en_text, x1, y1, x2, y2)

    _, buffer = cv2.imencode('.jpg', img_cv)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return jsonify({'processed_image': f"data:image/jpeg;base64,{img_str}"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)