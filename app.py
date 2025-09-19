# app.py
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from Backend.Controller.csv_handling import read_csv_preview, summarize_csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'Template')   # ← 根目录下的 Template
STATIC_DIR   = os.path.join(BASE_DIR, 'Static')     # ← 根目录下的 Static
UPLOAD_DIR   = os.path.join(BASE_DIR, '_uploads')

print("TEMPLATE_DIR:", TEMPLATE_DIR, "exists:", os.path.exists(TEMPLATE_DIR))
print("STATIC_DIR:", STATIC_DIR, "exists:", os.path.exists(STATIC_DIR))

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
ALLOWED_EXT = {'.csv'}

def _allowed(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT

@app.route('/')
def home():
    return render_template('Home.html', active_page='home')

@app.route('/csv')
def csv():
    return render_template('CSV.html', active_page='csv')
@app.route('/hormemo')
def hormemo():
    return render_template('hormemo.html', active_page='hormemo')
@app.route('/limit')
def limit():
    return render_template('limit.html', active_page='limit')
@app.post('/api/upload')
def api_upload():
    """接收CSV文件，返回预览与概要信息"""
    if 'file' not in request.files:
        return jsonify({'ok': False, 'error': 'no file field'}), 400
    f = request.files['file']
    if not f or f.filename.strip() == '':
        return jsonify({'ok': False, 'error': 'empty filename'}), 400
    if not _allowed(f.filename):
        return jsonify({'ok': False, 'error': 'only .csv allowed'}), 400

    filename = secure_filename(f.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    # 为了既能分析又能保存：先把文件读到内存再各用一份流
    data = f.read()

    try:
        preview = read_csv_preview(data, n=10)   # 前10行
        summary = summarize_csv(data)            # 行数/列数/缺失等
    except Exception as e:
        return jsonify({'ok': False, 'error': f'parse failed: {e}'}), 400

    # 保存原文件
    with open(save_path, 'wb') as out:
        out.write(data)

    return jsonify({
        'ok': True,
        'filename': filename,
        'saved_to': save_path,
        'preview': preview,       # list[dict]
        'summary': summary        # dict
    })

@app.errorhandler(413)
def too_large(_):
    return jsonify({'ok': False, 'error': 'file too large'}), 413

if __name__ == '__main__':
    app.run(debug=True)
