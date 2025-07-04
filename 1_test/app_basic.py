from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from datetime import datetime
from analyzer.room_analyzer import RoomAnalyzer

app = Flask(__name__)
CORS(app)

analyzer = RoomAnalyzer()

@app.route('/')
def index():
    return render_template('room_analyzer.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_room():
    try:
        data = request.get_json()
        image_data = data.get('image')
        reference_size = data.get('reference_size', 200)
        options = data.get('options', {})
        
        if not image_data:
            return jsonify({'success': False, 'error': '이미지가 없습니다.'})
        
        result = analyzer.analyze_image(image_data, reference_size, options)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    print("🚀 방 사진 분석 서버를 시작합니다...")
    print("📍 http://localhost:5000 에서 접속하세요")
    app.run(debug=True, host='0.0.0.0', port=5000)
