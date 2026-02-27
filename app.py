from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pytesseract
import base64

app = Flask(__name__)
CORS(app) # للسماح للواجهة بالتواصل مع السيرفر

# تنبيه: قم بتغيير هذا المسار إذا قمت بتثبيت Tesseract في مكان مختلف
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@app.route('/upload-id', methods=['POST'])
def process_id():
    if 'document' not in request.files:
        return jsonify({'error': 'لم يتم العثور على ملف'}), 400

    file = request.files['document']
    
    # 1. قراءة الصورة باستخدام OpenCV
    img_bytes = file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # 2. استخراج النص لمعرفة نوع الهوية (عربي + إنجليزي)
    custom_config = r'-l ara+eng --oem 3 --psm 6'
    text = pytesseract.image_to_string(img, config=custom_config)
    
    # تحديد نوع الهوية بشكل مبسط للنموذج الأولي
    doc_type = "وثيقة غير معروفة"
    if "وطنية" in text or "الوطنية" in text:
        doc_type = "بطاقة هوية وطنية"
    elif "جامعة" in text or "طالب" in text:
        doc_type = "بطاقة جامعية"

    # 3. التضليل التلقائي (محاكاة: تضليل أي أرقام تظهر في الهوية لحماية الخصوصية)
    d = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 40: # إذا كان الذكاء الاصطناعي واثقاً من الكلمة
            word = d['text'][i]
            # إذا كانت الكلمة تحتوي على أرقام (مثل الرقم الوطني أو المواليد)
            if any(char.isdigit() for char in word):
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                # رسم مستطيل أسود فوق الرقم لتضليله
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)

    # 4. تحويل الصورة المضللة لإرسالها للواجهة
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'doc_type': doc_type,
        'extracted_text': text[:100], # نرسل جزء من النص فقط للتأكيد
        'processed_image': f"data:image/jpeg;base64,{img_base64}"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
