"""
3D to Real Image Converter
2단계 AI 변환 웹 인터페이스
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from stage1_processor import StyleProcessor
from stage2_processor import PhotorealisticProcessor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 업로드 허용 파일 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# 프로세서 초기화
style_processor = StyleProcessor()
photorealistic_processor = PhotorealisticProcessor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """이미지 업로드"""
    if 'image' not in request.files:
        return jsonify({'error': '이미지가 선택되지 않았습니다'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다'}), 400
    
    if file and allowed_file(file.filename):
        # 고유 파일명 생성
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        extension = filename.rsplit('.', 1)[1].lower()
        new_filename = f"{file_id}.{extension}"
        
        # 업로드 폴더에 저장
        filepath = os.path.join('uploads', new_filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'filename': new_filename,
            'message': '이미지가 성공적으로 업로드되었습니다'
        })
    
    return jsonify({'error': '지원하지 않는 파일 형식입니다'}), 400

@app.route('/process/stage1', methods=['POST'])
def process_stage1():
    """1단계: 스타일 변경 (배치 유지)"""
    data = request.get_json()
    file_id = data.get('file_id')
    style = data.get('style', 'modern')
    
    if not file_id:
        return jsonify({'error': 'file_id가 필요합니다'}), 400
    
    try:
        # 업로드된 파일 찾기
        input_files = [f for f in os.listdir('uploads') if f.startswith(file_id)]
        if not input_files:
            return jsonify({'error': '업로드된 파일을 찾을 수 없습니다'}), 404
        
        input_path = os.path.join('uploads', input_files[0])
        
        # 1단계 처리 실행
        print(f"[STAGE1] 스타일 변경 시작: {style}")
        output_path = style_processor.process(input_path, style, file_id)
        
        return jsonify({
            'success': True,
            'stage1_output': output_path,
            'message': f'{style} 스타일 변경 완료'
        })
        
    except Exception as e:
        print(f"Stage1 오류: {e}")
        return jsonify({'error': f'1단계 처리 실패: {str(e)}'}), 500

@app.route('/process/stage2', methods=['POST'])
def process_stage2():
    """2단계: 실사화"""
    data = request.get_json()
    file_id = data.get('file_id')
    
    if not file_id:
        return jsonify({'error': 'file_id가 필요합니다'}), 400
    
    try:
        # 1단계 출력 파일 찾기
        stage1_files = [f for f in os.listdir('stage1_output') if f.startswith(file_id)]
        if not stage1_files:
            return jsonify({'error': '1단계 결과 파일을 찾을 수 없습니다'}), 404
        
        input_path = os.path.join('stage1_output', stage1_files[0])
        
        # 2단계 처리 실행
        print(f"[STAGE2] 실사화 시작")
        output_path = photorealistic_processor.process(input_path, file_id)
        
        return jsonify({
            'success': True,
            'stage2_output': output_path,
            'message': '실사화 완료'
        })
        
    except Exception as e:
        print(f"Stage2 오류: {e}")
        return jsonify({'error': f'2단계 처리 실패: {str(e)}'}), 500

@app.route('/process/full', methods=['POST'])
def process_full():
    """전체 처리 (1단계 + 2단계)"""
    data = request.get_json()
    file_id = data.get('file_id')
    style = data.get('style', 'modern')
    
    if not file_id:
        return jsonify({'error': 'file_id가 필요합니다'}), 400
    
    try:
        # 1단계 실행
        stage1_result = process_stage1_internal(file_id, style)
        if not stage1_result['success']:
            return jsonify(stage1_result), 500
        
        # 2단계 실행
        stage2_result = process_stage2_internal(file_id)
        if not stage2_result['success']:
            return jsonify(stage2_result), 500
        
        return jsonify({
            'success': True,
            'stage1_output': stage1_result['stage1_output'],
            'stage2_output': stage2_result['stage2_output'],
            'message': '2단계 변환 완료'
        })
        
    except Exception as e:
        print(f"Full process 오류: {e}")
        return jsonify({'error': f'전체 처리 실패: {str(e)}'}), 500

def process_stage1_internal(file_id, style):
    """내부용 1단계 처리"""
    try:
        input_files = [f for f in os.listdir('uploads') if f.startswith(file_id)]
        if not input_files:
            return {'success': False, 'error': '업로드된 파일을 찾을 수 없습니다'}
        
        input_path = os.path.join('uploads', input_files[0])
        output_path = style_processor.process(input_path, style, file_id)
        
        return {'success': True, 'stage1_output': output_path}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def process_stage2_internal(file_id):
    """내부용 2단계 처리"""
    try:
        stage1_files = [f for f in os.listdir('stage1_output') if f.startswith(file_id)]
        if not stage1_files:
            return {'success': False, 'error': '1단계 결과 파일을 찾을 수 없습니다'}
        
        input_path = os.path.join('stage1_output', stage1_files[0])
        output_path = photorealistic_processor.process(input_path, file_id)
        
        return {'success': True, 'stage2_output': output_path}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.route('/<path:filename>')
def serve_static_file(filename):
    """정적 파일 서빙 (이미지 표시용)"""
    if filename.startswith('stage1_output/') or filename.startswith('stage2_output/'):
        return send_file(filename)
    else:
        return jsonify({'error': '잘못된 파일 경로'}), 404

@app.route('/download/<path:filename>')
def download_file(filename):
    """파일 다운로드"""
    # 보안을 위해 파일 경로 검증
    if filename.startswith('stage1_output/'):
        return send_file(filename, as_attachment=True)
    elif filename.startswith('stage2_output/'):
        return send_file(filename, as_attachment=True)
    else:
        return jsonify({'error': '잘못된 파일 경로'}), 400

def main():
    """uv run으로 실행할 메인 함수"""
    # 필요한 폴더 생성
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('stage1_output', exist_ok=True)
    os.makedirs('stage2_output', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("3D to Real Image Converter 서버 시작")
    print("업로드 폴더 준비 완료")
    print("AI 프로세서 초기화 완료")
    print("http://localhost:5000 에서 접속 가능")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()