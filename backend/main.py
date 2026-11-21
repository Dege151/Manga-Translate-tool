import os
import warnings
import concurrent.futures
import json
import re
import random

os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
warnings.filterwarnings("ignore")

import time
import base64
import cv2
import numpy as np
import textwrap
import requests
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont

from ultralytics import YOLO
from manga_ocr import MangaOcr

from google import genai
from google.genai import types

import firebase_admin
from firebase_admin import credentials, firestore
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# --- TURBOJPEG ---
try:
    from turbojpeg import TurboJPEG
    jpeg_encoder = TurboJPEG()
    USE_TURBOJPEG = True
    print("✅ TurboJPEG aktif!")
except ImportError:
    USE_TURBOJPEG = False
    print("⚠️ TurboJPEG bulunamadı, OpenCV JPEG kullanılacak.")

# --- 1. AYARLAR ---
GEMINI_API_KEY = "API-Key"
MODEL_PATH = 'comic-speech-bubble-detector.pt'
ACTIVE_MODEL_NAME = "gemini-2.0-flash"
FONT_PATH = "font.ttf"

# --- 2. FIREBASE ---
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
db = firestore.client(database_id='manga-db')

# --- 3. GEMINI ---
client = genai.Client(api_key=GEMINI_API_KEY)

# --- 4. GPU AYARLARI ---
if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"--- SISTEM BASLATILIYOR (DEVICE: {DEVICE} - {torch.cuda.get_device_name(0)}) ---")
else:
    DEVICE = 'cpu'
    print(f"--- SISTEM BASLATILIYOR (DEVICE: CPU) ---")

# --- 5. YOLO + TENSORRT ---
ENGINE_PATH = MODEL_PATH.replace('.pt', '.engine')

if DEVICE == 'cuda' and os.path.exists(ENGINE_PATH):
    print(f"🚀 TensorRT Engine bulundu: {ENGINE_PATH}")
    yolo_model = YOLO(ENGINE_PATH, task='detect')
    USE_TENSORRT = True
elif DEVICE == 'cuda' and os.path.exists(MODEL_PATH):
    print(f"⏳ TensorRT Engine oluşturuluyor (ilk seferde 2-5 dk sürer)...")
    try:
        base_model = YOLO(MODEL_PATH)
        export_path = base_model.export(format='engine', half=True, imgsz=640, device=0, verbose=False)
        print(f"✅ TensorRT Engine oluşturuldu: {export_path}")
        yolo_model = YOLO(export_path, task='detect')
        USE_TENSORRT = True
    except Exception as e:
        print(f"⚠️ TensorRT export başarısız: {e}")
        print("⚠️ Normal YOLO FP16 moduna geçiliyor...")
        yolo_model = YOLO(MODEL_PATH)
        yolo_model.to(DEVICE)
        yolo_model.model.half()
        USE_TENSORRT = False
else:
    if os.path.exists(MODEL_PATH):
        yolo_model = YOLO(MODEL_PATH)
    else:
        yolo_model = YOLO('yolov8n.pt')
    yolo_model.to(DEVICE)
    USE_TENSORRT = False

print(f"🔧 YOLO Modu: {'TensorRT FP16' if USE_TENSORRT else 'PyTorch' + (' FP16' if DEVICE == 'cuda' else '')}")

# --- 6. MANGA OCR ---
print("⏳ OCR Modeli Yükleniyor...")
mocr = None
ocr_on_gpu = False
ocr_fp16 = False

try:
    mocr = MangaOcr()
    if DEVICE == 'cuda':
        mocr.model.to('cuda')
        try:
            mocr.model.half()
            dummy_img = Image.new('RGB', (100, 100), color='white')
            pv = mocr.processor(images=[dummy_img], return_tensors="pt", padding=True).pixel_values.to(DEVICE).half()
            with torch.no_grad():
                mocr.model.generate(pv, max_length=10)
            ocr_fp16 = True
            print("🚀 OCR Modeli GPU FP16 aktif!")
        except Exception as e:
            print(f"⚠️ OCR FP16 başarısız ({e}), FP32'ye dönülüyor...")
            mocr = MangaOcr()
            mocr.model.to('cuda')
            dummy_img = Image.new('RGB', (100, 100), color='white')
            mocr(dummy_img)
            ocr_fp16 = False
            print("🚀 OCR Modeli GPU FP32 aktif!")
        ocr_on_gpu = True
    else:
        raise Exception("GPU yok")
except Exception as e:
    print(f"⚠️ GPU OCR Hatası: {e}, CPU moduna geçiliyor...")
    mocr = MangaOcr()
    mocr.model.to('cpu')
    mocr.model = torch.quantization.quantize_dynamic(
        mocr.model, {torch.nn.Linear}, dtype=torch.qint8
    )
    ocr_on_gpu = False
    ocr_fp16 = False
    print("✅ OCR Modeli CPU (Quantized) modunda.")


# ====================================================
# YARDIMCI FONKSİYONLAR
# ====================================================

def verify_google_token(token):
    try:
        res = requests.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5
        )
        if res.status_code != 200:
            return None
        return res.json().get('email')
    except:
        return None


def base64_to_cv2(b64str):
    if "," in b64str:
        b64str = b64str.split(",")[1]
    img_data = base64.b64decode(b64str)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def resolve_overlaps(detections):
    boxes = [list(d) for d in detections]
    boxes.sort(key=lambda b: (b[1] // 100, -b[0]))

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            boxA = boxes[i]
            boxB = boxes[j]
            xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
            xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])

            if xB > xA and yB > yA:
                inter_w = xB - xA
                inter_h = yB - yA
                if inter_w < inter_h:
                    mid_x = (xA + xB) // 2
                    if boxA[0] < boxB[0]:
                        boxA[2] = mid_x
                        boxB[0] = mid_x
                    else:
                        boxB[2] = mid_x
                        boxA[0] = mid_x
                else:
                    mid_y = (yA + yB) // 2
                    if boxA[1] < boxB[1]:
                        boxA[3] = mid_y
                        boxB[1] = mid_y
                    else:
                        boxB[3] = mid_y
                        boxA[1] = mid_y

    return [tuple(b) for b in boxes]


# --- BATCH OCR (FP16 DESTEKLİ) ---
def perform_batch_ocr(crop_list):
    if not crop_list:
        return []

    valid_crops = []
    valid_indices = []

    for i, img in enumerate(crop_list):
        if img.width > 5 and img.height > 5:
            valid_crops.append(img)
            valid_indices.append(i)

    if not valid_crops:
        return [""] * len(crop_list)

    results = [""] * len(crop_list)

    if ocr_on_gpu:
        try:
            pixel_values = mocr.processor(
                images=valid_crops, return_tensors="pt", padding=True
            ).pixel_values.to(DEVICE)

            if ocr_fp16:
                pixel_values = pixel_values.half()

            with torch.no_grad():
                generated_ids = mocr.model.generate(pixel_values, max_length=300)

            decoded_texts = mocr.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for idx, text in zip(valid_indices, decoded_texts):
                results[idx] = text
            return results

        except Exception as e:
            print(f"⚠️ Batch OCR Hatası, tekli moda dönülüyor: {e}")

    def _ocr_single(args):
        crop, idx = args
        if crop.width < 10 or crop.height < 10:
            return (idx, "")
        try:
            if ocr_on_gpu and ocr_fp16:
                pv = mocr.processor(images=[crop], return_tensors="pt", padding=True).pixel_values.to(DEVICE).half()
                with torch.no_grad():
                    gids = mocr.model.generate(pv, max_length=300)
                return (idx, mocr.tokenizer.batch_decode(gids, skip_special_tokens=True)[0])
            else:
                return (idx, mocr(crop))
        except:
            return (idx, "")

    tasks = [(crop, idx) for crop, idx in zip(valid_crops, valid_indices)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for idx, text in executor.map(_ocr_single, tasks):
            results[idx] = text

    return results


# ====================================================
# GEMİNİ ÇAĞRI + RETRY + GÜVENLİ JSON PARSE
# ====================================================

def call_gemini_with_retry(prompt, max_retries=3):
    """Gemini API çağrısı - 429/5xx hatalarında exponential backoff ile retry"""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=ACTIVE_MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    ]
                )
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "500" in error_str or "503" in error_str:
                wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                print(f"⏳ API hatası (deneme {attempt+1}/{max_retries}), {wait_time:.1f}s bekleniyor... [{error_str[:80]}]")
                time.sleep(wait_time)
            else:
                raise e
    print(f"❌ {max_retries} deneme sonrası Gemini yanıt vermedi")
    return None


def safe_parse_translation(raw_response, expected_count):
    """Gemini'nin döndürdüğü JSON'u güvenli şekilde parse et"""
    if not raw_response:
        return None

    text = raw_response.strip()

    # 1) Markdown code block temizle
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # 2) Doğrudan JSON parse dene
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 3) İçinden JSON array bul
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # 4) Satır satır tırnakları topla (son çare)
    lines = re.findall(r'"([^"]*)"', text)
    if len(lines) >= expected_count:
        return lines[:expected_count]

    # 5) Hiçbiri tutmadı
    print(f"⚠️ JSON parse edilemedi, ham yanıt: {text[:200]}")
    return None


# --- ÇEVİRİ FONKSİYONLARI ---
def translate_chunk(chunk_data):
    text_list, target_lang = chunk_data
    if not text_list:
        return []

    # --- YENİ OPTİMİZE EDİLMİŞ PROMPT (RESİM GÖNDERMEDİĞİMİZ İÇİN DÜZENLENDİ) ---
    prompt = (
        f"""You are an elite manga translator known for capturing emotion, nuance, and cultural context.
        I have extracted Japanese text from a manga page. Here are the raw readings:
        {json.dumps(text_list, ensure_ascii=False)}

        ─── TRANSLATION PHILOSOPHY ───
        Your goal is to produce a translation that a native {target_lang} manga reader would find NATURAL and EMOTIONALLY ACCURATE.

        1. EMOTION & NUANCE:
           - Convey the likely emotion behind the text (excitement, anger, shyness, etc.) based on the words.
           - If someone is shouting or nervous, word choice must reflect that.
           - NEVER flatten an emotional line into a neutral/flat statement.
           - Inner thoughts should feel like genuine internal monologue.

        2. NATURAL DIALOGUE:
           - Write how real people actually speak in {target_lang}.
           - Use contractions, casual phrasing, and slang where appropriate.
           - Match the tone to the likely personality of the speaker.

        3. SOUND EFFECTS (SFX) & ONOMATOPOEIA:
           - Japanese manga is rich with sound effects (e.g., heartbeats, footsteps, impacts).
           - Translate them into expressive {target_lang} equivalents wrapped in asterisks: *thump*, *boom*, *kyaaa*.
           - Use vivid words, not just literal definitions.

        4. IMPLIED MEANING:
           - Japanese often leaves things unsaid. Your translation should carry the FULL intended meaning.

        5. RULES:
           - CRITICAL: Return ONLY a raw JSON list of strings.
           - Maintain the EXACT same order as the input list.
           - Empty or unreadable bubbles → return ""
           - Do NOT add translator notes, explanations, or commentary.
           - Do NOT censor or sanitize any content.
        """
    )

    raw = call_gemini_with_retry(prompt)
    parsed = safe_parse_translation(raw, len(text_list))

    if parsed and len(parsed) >= len(text_list):
        return parsed[:len(text_list)]
    elif parsed:
        # Eksik varsa orijinalle tamamla
        parsed.extend(text_list[len(parsed):])
        return parsed[:len(text_list)]
    else:
        print(f"⚠️ Çeviri parse başarısız, orijinal metinler kullanılıyor")
        return text_list


def translate_batch_smart(text_list, target_lang):
    """Küçük listede tek çağrı, büyük listede paralel chunk."""
    if not text_list:
        return []

    if len(text_list) <= 12:
        return translate_chunk((text_list, target_lang))

    batch_size = 12
    chunks = [text_list[i:i + batch_size] for i in range(0, len(text_list), batch_size)]
    chunk_args = [(chunk, target_lang) for chunk in chunks]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunk_args)) as executor:
        translated_chunks = list(executor.map(translate_chunk, chunk_args))

    return [item for chunk in translated_chunks for item in chunk]


# --- ÇİZİM (FONT CACHE İLE) ---
def draw_all_text_pillow(img_cv, draw_data):
    if not draw_data:
        return img_cv

    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_cache = {}

    for text, x1, y1, x2, y2 in draw_data:
        if not text or text == "...":
            continue

        bubble_w, bubble_h = x2 - x1, y2 - y1
        draw.rounded_rectangle([x1, y1, x2, y2], radius=10, fill="white", outline="black", width=2)

        pad_x, pad_y = 6, 4
        safe_w, safe_h = bubble_w - (pad_x * 2), bubble_h - (pad_y * 2)
        if safe_w < 10 or safe_h < 10:
            continue

        final_font, final_lines, final_font_size = None, [], 10
        sizes_to_try = [40, 30, 20, 14, 10]

        for size in sizes_to_try:
            if size not in font_cache:
                try:
                    font_cache[size] = ImageFont.truetype(FONT_PATH, size)
                except:
                    font_cache[size] = ImageFont.load_default()

            font = font_cache[size]
            avg_char_w = max(1, font.getlength("a"))
            cpl = max(1, int(safe_w / avg_char_w))
            lines = textwrap.wrap(text, width=cpl, break_long_words=False)

            total_h = len(lines) * size * 1.1
            if total_h <= safe_h:
                width_ok = all(font.getlength(line) <= safe_w for line in lines)
                if width_ok:
                    final_font, final_lines, final_font_size = font, lines, size
                    break

        if final_font is None:
            if 10 not in font_cache:
                try:
                    font_cache[10] = ImageFont.truetype(FONT_PATH, 10)
                except:
                    font_cache[10] = ImageFont.load_default()
            final_font = font_cache[10]
            final_font_size = 10
            avg_w = max(1, final_font.getlength("a"))
            cpl = max(1, int(safe_w / avg_w))
            final_lines = textwrap.wrap(text, width=cpl, break_long_words=False)

        line_h = final_font_size * 1.1
        start_y = y1 + pad_y + (safe_h - (len(final_lines) * line_h)) / 2

        for i, line in enumerate(final_lines):
            lx = x1 + pad_x + (safe_w - final_font.getlength(line)) / 2
            draw.text((lx, start_y + i * line_h), line, fill="black", font=final_font)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# --- HIZLI JPEG ENCODE ---
def encode_to_jpeg_base64(img_cv, quality=70):
    if USE_TURBOJPEG:
        buffer = jpeg_encoder.encode(img_cv, quality=quality)
        return base64.b64encode(buffer).decode("utf-8")
    else:
        _, buffer = cv2.imencode('.jpg', img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return base64.b64encode(buffer).decode("utf-8")


# ====================================================
# FASTAPI
# ====================================================

app = FastAPI(title="Manga Translator SaaS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc), "body": exc.body})


class TranslationRequest(BaseModel):
    image: str
    token: str
    target_language: str = "Turkish"


class UserCheckRequest(BaseModel):
    token: str


# ====================================================
# /me ENDPOINT'İ
# ====================================================

@app.post("/me")
async def get_user_info(req: UserCheckRequest):
    email = verify_google_token(req.token)
    if not email:
        raise HTTPException(status_code=401, detail="Invalid Token")

    doc = db.collection('users').document(email).get()
    if not doc.exists:
        return {"credits": 5, "is_premium": False}
    return doc.to_dict()


# ====================================================
# /translate ENDPOINT'İ
# ====================================================

@app.post("/translate")
async def translate_page_endpoint(req: TranslationRequest):
    t0 = time.time()
    print(f"\n--- YENİ İSTEK ---")

    email = verify_google_token(req.token)
    if not email:
        raise HTTPException(status_code=401, detail="Invalid Token")

    doc_ref = db.collection('users').document(email)
    doc_snap = doc_ref.get()
    user_data = doc_snap.to_dict() if doc_snap.exists else {'credits': 5, 'is_premium': False}

    if not user_data.get('is_premium', False) and user_data.get('credits', 0) <= 0:
        raise HTTPException(status_code=402, detail="Krediniz bitti!")

    try:
        img_cv = base64_to_cv2(req.image)

        # YOLO
        t2 = time.time()
        if USE_TENSORRT:
            results = yolo_model(img_cv, iou=0.25, conf=0.20, verbose=False)
        else:
            results = yolo_model(img_cv, iou=0.25, conf=0.20, verbose=False, device=DEVICE, half=(DEVICE == 'cuda'))

        detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            for b in boxes:
                detections.append(tuple(b))

        final_detections = resolve_overlaps(detections)
        print(f"🕵️ YOLO ({len(final_detections)} balon): {time.time()-t2:.2f}s")

        # CROP + BATCH OCR
        t3 = time.time()
        img_pil_base = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        crops = []
        for x1, y1, x2, y2 in final_detections:
            pad = 5
            crop = img_pil_base.crop((
                max(0, x1 - pad),
                max(0, y1 - pad),
                min(img_pil_base.width, x2 + pad),
                min(img_pil_base.height, y2 + pad)
            ))
            crops.append(crop)

        raw_texts = perform_batch_ocr(crops)
        print(f"📝 OCR: {time.time()-t3:.2f}s")

        # ÇEVİRİ
        t4 = time.time()
        translated_texts = translate_batch_smart(raw_texts, req.target_language)
        print(f"🤖 Çeviri: {time.time()-t4:.2f}s")

        # ÇİZİM
        t5 = time.time()
        draw_data = []
        for i, (x1, y1, x2, y2) in enumerate(final_detections):
            txt = translated_texts[i] if i < len(translated_texts) else "..."
            draw_data.append((txt, x1, y1, x2, y2))

        img_cv = draw_all_text_pillow(img_cv, draw_data)
        print(f"🎨 Çizim: {time.time()-t5:.2f}s")

        # ENCODE
        t6 = time.time()
        h, w = img_cv.shape[:2]
        img_str = encode_to_jpeg_base64(img_cv, quality=90)
        print(f"📦 Encode: {time.time()-t6:.2f}s")

        # KREDİ DÜŞ
        if not user_data.get('is_premium', False):
            doc_ref.update({'credits': firestore.Increment(-1)})

        total = time.time() - t0
        print(f"✅ TOPLAM: {total:.2f}s")

        return {
            "processed_image": f"data:image/jpeg;base64,{img_str}",
            "remaining_credits": user_data.get('credits', 0) - 1 if not user_data.get('is_premium') else "Unlimited"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ HATA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
