import os
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_pir_apply_shape_optimization_pass"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import cv2
import numpy as np
import logging
import base64
import json
import re
from PIL import Image
from google import genai
from google.genai import types

logging.getLogger("ppocr").setLevel(logging.ERROR)
from paddleocr import PaddleOCR

# 🔴 إعداد مفتاح جوجل
client = genai.Client(api_key="ضع_مفتاحك_هنا")

app = Flask(__name__)
CORS(app)

# ── إعدادات قاعدة البيانات (SQLite) ──
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///wathiqati.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ── تصميم جدول البيانات (Database Model) ──
class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(200), nullable=False)    # اسم صاحب الوثيقة
    doc_type = db.Column(db.String(100), nullable=False)     # نوع الوثيقة
    finder_name = db.Column(db.String(100))                  # اسم الواجد (من الواجهة)
    contact_info = db.Column(db.String(100))                 # طريقة التواصل (من الواجهة)
    location_found = db.Column(db.String(200))               # مكان العثور (من الواجهة)
    processed_image = db.Column(db.Text, nullable=False)     # الصورة المضللة (Base64)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow) # وقت الرفع

# إنشاء ملف قاعدة البيانات تلقائياً عند التشغيل
with app.app_context():
    db.create_all()

print("جاري تشغيل محرك PaddleOCR ومحرك Gemini وقاعدة البيانات... 🚀")
ocr = PaddleOCR(use_textline_orientation=False, lang='ar', device='cpu')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("✅ السيرفر الهجين متصل بقاعدة البيانات وجاهز للعمل!")

# ── دوال التضليل الخاصة بك (نفسها بدون تغيير) ──
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
    cv2.rectangle(image, (max(0, x_min - pad_x), max(0, y_min - pad_y)), (min(image.shape[1], x_max + pad_x), min(image.shape[0], y_max + pad_y)), (15, 15, 15), -1)

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
                for item in result: ocr_data.append(item)
    except:
        try:
            results = ocr.ocr(img)
            if results and results[0]:
                for item in results[0]: ocr_data.append(item)
        except Exception as e: pass
    return ocr_data

def contains_any(text, keywords):
    return any(k in text.strip() for k in keywords)


# ── المسار الأول: رفع الوثيقة وحفظها ──
@app.route('/upload-id', methods=['POST'])
def process_id():
    if 'document' not in request.files:
        return jsonify({'error': 'لم يتم العثور على ملف'}), 400

    # استلام بيانات الفورم من الواجهة (بيانات الواجد)
    finder_name = request.form.get('finderName', 'فاعل خير')
    contact_info = request.form.get('contactInfo', 'غير محدد')
    location_found = request.form.get('location', 'غير محدد')

    file = request.files['document']
    np_img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    h_img, w_img = img.shape[:2]

    # (تضليل الوجه)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
    for (x, y, w, h) in faces:
        img[y:y+h, x:x+w] = cv2.GaussianBlur(img[y:y+h, x:x+w], (121, 121), 50)

    # (تضليل النصوص بـ PaddleOCR)
    ocr_data = run_ocr(img)
    lines = []
    for line_info in ocr_data:
        try:
            bbox, (text, prob) = line_info
            xs, ys = [p[0] for p in bbox], [p[1] for p in bbox]
            x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

            placed = False
            for line in lines:
                if abs(line['y_mid'] - cy) < 22:
                    line['items'].append({'text': text, 'cx': cx, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max})
                    placed = True
                    break
            if not placed:
                lines.append({'y_mid': cy, 'items': [{'text': text, 'cx': cx, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}]})
        except: continue

    for line in lines:
        line['items'].sort(key=lambda x: x['cx'], reverse=True)
        line['txt'] = " ".join(i['text'] for i in line['items'])

    mother_line = mother_gdpa_line = blood_line = gender_line = natnum_line = smallnum_line = None
    passed_mother = False

    for line in sorted(lines, key=lambda x: x['y_mid']):
        txt = line['txt']
        if line['y_mid'] < h_img * 0.18: continue
        if contains_any(txt, KEYWORDS['mother']): mother_line = line; passed_mother = True
        elif passed_mother and contains_any(txt, KEYWORDS['grandpa']) and not mother_gdpa_line: mother_gdpa_line = line
        if contains_any(txt, KEYWORDS['blood']): blood_line = line
        if contains_any(txt, KEYWORDS['gender']): gender_line = line
        for item in line['items']:
            digits_only = re.sub(r'\D', '', item['text'])
            if len(digits_only) >= 8 and not natnum_line: natnum_line = line
            if re.search(r'[A-Za-z]{1,3}\d{4,}', item['text']) and not smallnum_line: smallnum_line = line

    for line_info in ocr_data:
        try:
            bbox, (text, prob) = line_info
            digits = re.sub(r'\D', '', text)
            if len(digits) >= 6 or re.search(r'[A-Za-z]{1,3}\d{4,}', text):
                xs, ys = [p[0] for p in bbox], [p[1] for p in bbox]
                cv2.rectangle(img, (max(0, int(min(xs))-8), max(0, int(min(ys))-6)), (min(w_img, int(max(xs))+8), min(h_img, int(max(ys))+6)), (15, 15, 15), -1)
        except: continue

    for target in [mother_line, mother_gdpa_line, blood_line, gender_line, natnum_line, smallnum_line]:
        if target: redact_line_obj(img, target)

    # (استخراج البيانات بـ Gemini)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    prompt = """أنت خبير في المستمسكات الرسمية العراقية. استخرج المعلومات من الصورة. تجاهل المربعات السوداء واللغة غير العربية. أرجع النتيجة JSON فقط: {"doc_type": "نوع الوثيقة", "full_name": "الاسم الثلاثي كامل"}"""
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, pil_img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", temperature=0.1,
                safety_settings=[
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                ]
            )
        )
        extracted_data = json.loads(response.text)
        doc_type = extracted_data.get("doc_type", "وثيقة غير معروفة")
        full_name = extracted_data.get("full_name", "غير متوفر")
    except Exception as e:
        doc_type, full_name = "تعذر التحليل", "خطأ في الاستخراج"

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # 💾 الحفظ في قاعدة البيانات 💾
    if full_name != "خطأ في الاستخراج" and full_name != "غير متوفر":
        new_doc = Document(
            full_name=full_name,
            doc_type=doc_type,
            finder_name=finder_name,
            contact_info=contact_info,
            location_found=location_found,
            processed_image=f"data:image/jpeg;base64,{img_base64}"
        )
        db.session.add(new_doc)
        db.session.commit()
        print(f"✅ تم حفظ وثيقة ({full_name}) في قاعدة البيانات!")

    return jsonify({
        'full_name': full_name,
        'doc_type': doc_type,
        'processed_image': f"data:image/jpeg;base64,{img_base64}"
    })

# ── المسار الثاني: محرك البحث (للباحثين عن وثائقهم المفقودة) ──
@app.route('/search-id', methods=['GET'])
def search_id():
    # الحصول على الاسم المبحوث عنه من الرابط (مثال: /search-id?name=محمد)
    query_name = request.args.get('name', '').strip()
    
    if not query_name:
        return jsonify({'error': 'يرجى إدخال اسم للبحث'}), 400

    # البحث في قاعدة البيانات عن أي اسم يحتوي على الكلمة المبحوثة
    results = Document.query.filter(Document.full_name.like(f"%{query_name}%")).all()
    
    documents_list = []
    for doc in results:
        documents_list.append({
            'id': doc.id,
            'full_name': doc.full_name,
            'doc_type': doc.doc_type,
            'finder_name': doc.finder_name,
            'contact_info': doc.contact_info,
            'location_found': doc.location_found,
            'upload_date': doc.upload_date.strftime("%Y-%m-%d"),
            'processed_image': doc.processed_image
        })

    return jsonify({'results': documents_list, 'count': len(documents_list)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
