# app.py
import os
from flask import Flask, render_template
from Backend.Controller.csvcontroller import bp as csv_bp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'Template')
STATIC_DIR = os.path.join(BASE_DIR, 'Static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# 注册蓝图
app.register_blueprint(csv_bp)

@app.route('/')
def home():
    return render_template('Home.html', active_page='home')

@app.route('/csv')
def csv():
    return render_template('CSV.html', active_page='csv')

@app.route('/hormemo')
def hormemo():
    return render_template('hormemo.html', active_page='hormemo')

@app.errorhandler(413)
def too_large(_):
    return {'ok': False, 'error': 'file too large'}, 413

if __name__ == '__main__':
    app.run(debug=True)