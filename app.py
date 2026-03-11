import os
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_pir_apply_shape_optimization_pass"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import logging
import base64
import json
import re
from PIL import Image

# استيراد مكتبات جوجل الجديدة
from google import genai
from google.genai import types

logging.getLogger("ppocr").setLevel(logging.ERROR)
from paddleocr import PaddleOCR

# 🔴 إعداد مفتاح جوجل الخاص بك هنا
client = genai.Client(api_key="AIzaSyCOgoy2yOL_B5d-t5ox4EzQS7bRmxVZQbI")

app = Flask(__name__)
CORS(app)

print("جاري تشغيل محرك PaddleOCR (للتضليل) ومحرك Gemini (للاستخراج)... 🚀")
# الإبقاء على إعداداتك المستقرة تماماً لتجنب أي أخطاء في الـ CMD
ocr = PaddleOCR(use_textline_orientation=False, lang='ar', device='cpu')
print("تم تحميل الموديلات بنجاح! السيرفر الهجين الاحترافي جاهز للعمل 🚀")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# قواميسك الخاصة لتحديد أماكن التضليل بدقة
KEYWORDS = {
    'mother':  ['الأم', 'الام', 'مألا', 'دايك', 'كياد', 'داێك', 'تهده'],
    'grandpa': ['الجد', 'دجلا', 'باپير', 'ريياب', 'ريباب', 'باپێر'],
    'blood':   ['فصيلة', 'ةليصف', 'فصيله', 'الدم', 'مدلا', 'كروبي', 'يبورك', 'خوين', 'نيوخ', 'خوێن', 'O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB'],
    'gender':  ['الجنس', 'سنجلا', 'ذكر', 'ركذ', 'انثى', 'ىثنا', 'أنثى', 'رةگەز', 'زهكدر'],
}

def redact_line_obj(image, line_obj, pad_x=20, pad_y=10):
    if not line_obj or not line_obj['items']: return
    x_min = int(min(item['x_min'] for item in line_obj['items']))
    x_max = int(max(item['x_max'] for item in line_obj['items']))
    y_min = int(min(item['y_min'] for item in line_obj['items']))
    y_max = int(max(item['y_max'] for item in line_obj['items']))
    cv2.rectangle(image,
                  (max(0, x_min - pad_x), max(0, y_min - pad_y)),
                  (min(image.shape[1], x_max + pad_x), min(image.shape[0], y_max + pad_y)),
                  (15, 15, 15), -1)

def run_ocr(img):
    ocr_data = []
    try:
        results = list(ocr.predict(img))
        for result in results:
            if hasattr(result, 'rec_texts'):
                for bbox, text, score in zip(result.det_polygons, result.rec_texts, result.rec_scores):
                    pts = [[bbox[j], bbox[j+1]] for j in range(0, len(bbox), 2)]
                    ocr_data.append([pts, (text, float(score))])
            elif isinstance(result, list):
                for item in result:
                    ocr_data.append(item)
    except:
        try:
            results = ocr.ocr(img)
            if results and results[0]:
                for item in results[0]:
                    ocr_data.append(item)
        except Exception as e:
            print(f"OCR فشل: {e}")
    return ocr_data

def contains_any(text, keywords):
    t = text.strip()
    return any(k in t for k in keywords)


@app.route('/upload-id', methods=['POST'])
def process_id():
    if 'document' not in request.files:
        return jsonify({'error': 'لم يتم العثور على ملف'}), 400

    file = request.files['document']
    img_bytes = file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    h_img, w_img = img.shape[:2]

    # ── 1. تضليل الوجوه محلياً ──────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
    for (x, y, w, h) in faces:
        img[y:y+h, x:x+w] = cv2.GaussianBlur(img[y:y+h, x:x+w], (121, 121), 50)

    # ── 2. قراءة النصوص لتحديد الإحداثيات ───────────────────────────────
    ocr_data = run_ocr(img)

    # ── 3. تجميع الأسطر (نفس خوارزميتك الناجحة) ─────────────────────────
    lines = []
    for line_info in ocr_data:
        try:
            bbox, (text, prob) = line_info
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

            placed = False
            for line in lines:
                if abs(line['y_mid'] - cy) < 22:
                    line['items'].append({
                        'text': text, 'cx': cx,
                        'x_min': x_min, 'x_max': x_max,
                        'y_min': y_min, 'y_max': y_max,
                    })
                    placed = True
                    break
            if not placed:
                lines.append({'y_mid': cy, 'items': [{
                    'text': text, 'cx': cx,
                    'x_min': x_min, 'x_max': x_max,
                    'y_min': y_min, 'y_max': y_max,
                }]})
        except:
            continue

    for line in lines:
        line['items'].sort(key=lambda x: x['cx'], reverse=True)
        line['txt'] = " ".join(i['text'] for i in line['items'])

    # ── 4. البحث عن الأسطر الحساسة للتضليل ────────────────────────────
    mother_line = mother_gdpa_line = blood_line = None
    gender_line = natnum_line = smallnum_line = None
    passed_mother = False

    for line in sorted(lines, key=lambda x: x['y_mid']):
        txt = line['txt']
        if line['y_mid'] < h_img * 0.18:
            continue

        if contains_any(txt, KEYWORDS['mother']):
            mother_line = line
            passed_mother = True
        elif passed_mother and contains_any(txt, KEYWORDS['grandpa']) and not mother_gdpa_line:
            mother_gdpa_line = line

        if contains_any(txt, KEYWORDS['blood']):
            blood_line = line

        if contains_any(txt, KEYWORDS['gender']):
            gender_line = line

        for item in line['items']:
            digits_only = re.sub(r'\D', '', item['text'])
            if len(digits_only) >= 8 and not natnum_line:
                natnum_line = line
            if re.search(r'[A-Za-z]{1,3}\d{4,}', item['text']) and not smallnum_line:
                smallnum_line = line

    # ── 5. تضليل الأرقام المباشر ─────────────────────────────────────────
    for line_info in ocr_data:
        try:
            bbox, (text, prob) = line_info
            digits = re.sub(r'\D', '', text)
            if len(digits) >= 6 or re.search(r'[A-Za-z]{1,3}\d{4,}', text):
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                cv2.rectangle(img,
                    (max(0, int(min(xs))-8), max(0, int(min(ys))-6)),
                    (min(w_img, int(max(xs))+8), min(h_img, int(max(ys))+6)),
                    (15, 15, 15), -1)
        except:
            continue

    # ── 6. تضليل الأسطر الحساسة ──────────────────────────────────────────
    for target in [mother_line, mother_gdpa_line, blood_line, gender_line, natnum_line, smallnum_line]:
        if target:
            redact_line_obj(img, target)

    # ── 7. إرسال الصورة "المضللة" إلى الذكاء السحابي (Gemini) لاستخراج البيانات ──
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    prompt = """
    أنت خبير في قراءة المستمسكات الرسمية العراقية.
    استخرج المعلومات من هذه الصورة. تجاهل المربعات السوداء تماماً وتجاهل أي لغة غير العربية.
    أرجع النتيجة بصيغة JSON فقط بهذا الشكل:
    {"doc_type": "نوع الوثيقة (مثال: البطاقة الوطنية)", "full_name": "الاسم الثلاثي كامل (الاسم واسم الأب واسم الجد)"}
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, pil_img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        extracted_data = json.loads(response.text)
        
        doc_type = extracted_data.get("doc_type", "وثيقة غير معروفة")
        full_name = extracted_data.get("full_name", "غير متوفر")
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        doc_type = "تعذر التحليل السحابي"
        full_name = "حدث خطأ في قراءة الاسم"

    # تحويل الصورة النهائية لإرسالها للواجهة
    _, buffer = cv2.imencode('.jpg', img)
    
    return jsonify({
        'full_name': full_name,
        'doc_type': doc_type,
        'processed_image': f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
