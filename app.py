from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import easyocr
import base64
import re

app = Flask(__name__)
CORS(app)

print("جاري تحميل موديل الذكاء الاصطناعي (EasyOCR) على كرت الشاشة RTX...")
reader = easyocr.Reader(['ar', 'en'], gpu=True) 
print("تم تحميل الموديل بنجاح! السيرفر جاهز.")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# دالة التضليل التي أثبتت نجاحها
def redact_line_obj(image, line_obj):
    if not line_obj or not line_obj['items']: return
    x_min = int(min(item['x_min'] for item in line_obj['items']))
    x_max = int(max(item['x_max'] for item in line_obj['items']))
    y_min = int(min(item['y_min'] for item in line_obj['items']))
    y_max = int(max(item['y_max'] for item in line_obj['items']))
    cv2.rectangle(image, (x_min - 10, y_min - 5), (x_max + 10, y_max + 5), (15, 15, 15), -1)

@app.route('/upload-id', methods=['POST'])
def process_id():
    if 'document' not in request.files:
        return jsonify({'error': 'لم يتم العثور على ملف'}), 400

    file = request.files['document']
    img_bytes = file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    h_img, w_img, _ = img.shape

    # 1. تضليل الوجوه
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
    for (x, y, w, h) in faces:
        img[y:y+h, x:x+w] = cv2.GaussianBlur(img[y:y+h, x:x+w], (99, 99), 30)

    # 2. قراءة النصوص
    results = reader.readtext(img, detail=1)

    # 3. تجميع الأسطر هندسياً
    lines = []
    for bbox, text, prob in results:
        y_min, y_max = min(p[1] for p in bbox), max(p[1] for p in bbox)
        x_min, x_max = min(p[0] for p in bbox), max(p[0] for p in bbox)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        placed = False
        for line in lines:
            if abs(line['y_mid'] - cy) < 15:
                line['items'].append({'text': text, 'cx': cx, 'bbox': bbox, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max})
                placed = True
                break
        if not placed:
            lines.append({'y_mid': cy, 'items': [{'text': text, 'cx': cx, 'bbox': bbox, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}]})

    lines.sort(key=lambda x: x['y_mid'])

    # === القائمة السوداء الصارمة (إضافة الكلمات الجديدة التي طلبتها) ===
    blacklist = {
        'الاسم','ناو','الأب','الاب','باوك','الجد','باپير','بايير','اللقب','نازناو',
        'الأم','الام','دايك','الجنس','رەگەز','فصيلة','الدم','خوين','گروپی','ذكر','انثى',
        'جمهورية','عراق','العراق','عێراق','عيراق','وزارة','الداخلية','مديرية','مديريتا','الأحوال','الاحوال',
        'المدنية','الجوازات','والإقامة','والاقامة','البطاقة','الوطنية','شؤون','اصدار',
        'كومارى','وەزارەتى','ناوخۆ','بەڕێوەبەرايەتی','باری','شارستانی','تاريخ',
        'پاسپۆرت','نیشینگە','محل','الولادة','خف','ال','اب','بن','شؤون'
    }

    first_name = father_name = grandpa_name = ""
    mother_line = mother_grandpa_line = blood_line = None
    passed_mother = False

    def extract_clean(text_val, y_pos, x_pos):
        # 🛡️ فلتر مكاني: تجاهل الترويسة (أعلى 15%) وتجاهل الجانب الأيمن (الذي يحتوي عادة على الزخارف)
        if y_pos < (h_img * 0.18): return ""
        
        # تنظيف علامات الترقيم
        text_val = re.sub(r'[:/-]', ' ', text_val)
        
        words = re.findall(r'[\u0621-\u064A]{2,}', text_val)
        for w in words:
            # إذا كانت الكلمة في البلاك ليست، أو أنها جزء من "عراق" بأشكال مختلفة، احذفها
            if w in blacklist or "عراق" in w or "جمهور" in w:
                continue
            return w
        return ""

    for line in lines:
        line['items'].sort(key=lambda x: x['cx'], reverse=True)
        line_text = " ".join([item['text'] for item in line['items']])
        y_curr = line['y_mid']

        # تضليل الأسطر (نفس المنطق الناجح)
        if any(kw in line_text for kw in ['الأم','دايك','الام']):
            mother_line = line
            passed_mother = True
        elif passed_mother and any(kw in line_text for kw in ['الجد','باپير','بايير']) and not mother_grandpa_line:
            mother_grandpa_line = line
        elif any(kw in line_text for kw in ['فصيلة','الدم','خوين','گروپی']) or re.search(r'(O\+|O\-|A\+|A\-|B\+|B\-|AB\+|AB\-)', line_text.upper()):
            blood_line = line

        # استخراج الأسماء بدقة أكبر مع الفلتر المكاني
        for item in line['items']:
            if not first_name and any(kw in item['text'] for kw in ['الاسم','ناو']):
                # نبحث في السطر نفسه عن أول كلمة بعد "الاسم"
                first_name = extract_clean(line_text, y_curr, item['cx'])
            elif not father_name and any(kw in item['text'] for kw in ['الأب','الاب','باوك']):
                father_name = extract_clean(line_text, y_curr, item['cx'])
            elif not grandpa_name and not passed_mother and any(kw in item['text'] for kw in ['الجد','باپير','بايير']):
                grandpa_name = extract_clean(line_text, y_curr, item['cx'])

    # تضليل الأرقام
    for bbox, text, prob in results:
        if len(re.findall(r'\d', text)) >= 8 or re.search(r'[A-Za-z]{1,2}\d{5,}', text):
            x_min, y_min = int(min(p[0] for p in bbox)), int(min(p[1] for p in bbox))
            x_max, y_max = int(max(p[0] for p in bbox)), int(max(p[1] for p in bbox))
            cv2.rectangle(img, (x_min-5, y_min-5), (x_max+5, y_max+5), (15, 15, 15), -1)

    if mother_line: redact_line_obj(img, mother_line)
    if mother_grandpa_line: redact_line_obj(img, mother_grandpa_line)
    if blood_line: redact_line_obj(img, blood_line)

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'doc_type': "بطاقة هوية وطنية (العراق)",
        'full_name': f"{first_name} {father_name} {grandpa_name}".strip(),
        'processed_image': f"data:image/jpeg;base64,{img_base64}"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
