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
print("تم تحميل الموديل بنجاح! السيرفر جاهز للعمل بسرعة فائقة.")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# دالة ذكية لحساب منتصف الكلمة عمودياً
def get_y_center(bbox):
    return (min(p[1] for p in bbox) + max(p[1] for p in bbox)) / 2

# الخوارزمية الجديدة: تضليل السطر بالكامل (من أقصى اليمين لأقصى اليسار)
def redact_whole_line(image, results, target_y, y_tolerance=15):
    # جمع كل الكلمات التي تقع على نفس السطر
    line_boxes = [bbox for bbox, text, prob in results if abs(get_y_center(bbox) - target_y) < y_tolerance]
    if line_boxes:
        # حساب أبعد نقطة يميناً ويساراً في هذا السطر
        x_min = int(min(p[0] for bbox in line_boxes for p in bbox))
        x_max = int(max(p[0] for bbox in line_boxes for p in bbox))
        y_min = int(min(p[1] for bbox in line_boxes for p in bbox))
        y_max = int(max(p[1] for bbox in line_boxes for p in bbox))
        # رسم مستطيل أسود صلب يغطي كامل السطر بزيادة 8 بكسل للضمان
        cv2.rectangle(image, (x_min - 8, y_min - 8), (x_max + 8, y_max + 8), (15, 15, 15), -1)

@app.route('/upload-id', methods=['POST'])
def process_id():
    if 'document' not in request.files:
        return jsonify({'error': 'لم يتم العثور على ملف'}), 400

    file = request.files['document']
    img_bytes = file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. تضليل الوجوه
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    for (x, y, w, h) in faces:
        face_roi = img[y:y+h, x:x+w]
        face_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)
        img[y:y+h, x:x+w] = face_roi

    # 2. قراءة النصوص
    results = reader.readtext(gray, detail=1)
    
    # 3. دالة استخراج الأسماء بالترتيب الصحيح (للواجهة الأمامية)
    def find_value_after(keyword_ar, keyword_ku, start_idx=0):
        for i in range(start_idx, len(results)):
            text = results[i][1]
            if keyword_ar in text or keyword_ku in text:
                for j in range(i, min(i+4, len(results))):
                    val = results[j][1]
                    val = re.sub(r'[:/]', ' ', val).strip()
                    parts = val.split()
                    for p in parts:
                        if p not in ['الاسم','ناو','الأب','باوك','الجد','باپير','الأم','دايك','فصيلة','الدم','خوين','اللقب','نازناو'] and len(p)>1 and not any(c.isdigit() for c in p):
                            return p, j
        return "", start_idx

    first, idx = find_value_after('الاسم', 'ناو', 0)
    father, idx = find_value_after('الأب', 'باوك', idx)
    grandpa, idx = find_value_after('الجد', 'باپير', idx)
    mother, idx = find_value_after('الأم', 'دايك', idx)
    mother_grandpa, idx = find_value_after('الجد', 'باپير', idx)

    # 4. التضليل الشامل والدقيق للأسطر
    mother_y = None
    mother_grandpa_y = None
    blood_y = None
    
    passed_mother = False
    
    for i, (bbox, text, prob) in enumerate(results):
        # 4.1 تضليل الأرقام الطويلة (الرقم الوطني والعائلي)
        if len(re.findall(r'\d', text)) >= 8 or re.search(r'[A-Za-z]{1,2}\d{5,}', text):
            x_min = int(min(p[0] for p in bbox))
            y_min = int(min(p[1] for p in bbox))
            x_max = int(max(p[0] for p in bbox))
            y_max = int(max(p[1] for p in bbox))
            cv2.rectangle(img, (x_min-5, y_min-5), (x_max+5, y_max+5), (15, 15, 15), -1)

        # 4.2 تحديد السطور المراد إخفاؤها بالكامل (حفظ الإحداثيات)
        if 'الأم' in text or 'دايك' in text:
            mother_y = get_y_center(bbox)
            passed_mother = True
        elif passed_mother and ('الجد' in text or 'باپير' in text) and mother_grandpa_y is None:
            mother_grandpa_y = get_y_center(bbox) # الجد الذي يأتي بعد الأم هو جد الأم
        elif 'فصيلة' in text or 'الدم' in text or 'خوين' in text or re.search(r'^(O\+|O\-|A\+|A\-|B\+|B\-|AB\+|AB\-)$', text.upper().strip()):
            if blood_y is None:
                blood_y = get_y_center(bbox)

    # تطبيق التضليل على السطور المكتشفة بالكامل
    if mother_y is not None:
        redact_whole_line(img, results, mother_y)
    if mother_grandpa_y is not None:
        redact_whole_line(img, results, mother_grandpa_y)
    if blood_y is not None:
        redact_whole_line(img, results, blood_y)

    # تحديد النوع
    doc_type = "وثيقة غير معروفة"
    if any(k in res[1] for res in results for k in ["وطني", "عراق", "البطاقة", "الداخلية"]):
        doc_type = "بطاقة هوية وطنية (العراق)"

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'doc_type': doc_type,
        'names': {
            'first': first if first else "غير معروف",
            'father': father if father else "",
            'grandpa': grandpa if grandpa else "",
            'mother': mother if mother else "غير معروف",
            'mother_grandpa': mother_grandpa if mother_grandpa else ""
        },
        'processed_image': f"data:image/jpeg;base64,{img_base64}"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
