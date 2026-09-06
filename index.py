from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
CORS(app)

@app.route('/remove', methods=['POST'])
def remove_watermark():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.dilate(edges, kernel, iterations=2)
        result2 = cv2.inpaint(img, edges, 5, cv2.INPAINT_TELEA)
        
        final = cv2.addWeighted(result, 0.6, result2, 0.4, 0)
        final = cv2.convertScaleAbs(final, alpha=1.03, beta=3)
        
        _, buffer = cv2.imencode('.jpg', final)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{result_base64}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Vercel needs this
def handler(request, context):
    return app(request.environ, context)