from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import sys
import threading
import queue
import io
from contextlib import redirect_stdout
import model_12_final

app = Flask(__name__)
socketio = SocketIO(app)

# 创建队列用于存储输入
input_queue = queue.Queue()
# 创建事件用于同步
input_ready = threading.Event()

# 自定义输入输出处理类
class WebIO:
    def __init__(self):
        self.encoding = 'utf-8'
        # 添加需要过滤的文本列表
        self.filter_texts = [
            "* Serving Flask app",
            "* Debug mode:",
            "* Running on",
            "Press CTRL+C",
            "* Restarting with",
            "* Detected change",
            "WARNING: This is a development server.",
            "DO NOT USE IT IN A PRODUCTION DEPLOYMENT.",
            "WARNING:werkzeug:",
            "wrappers.py found.",
            "GenRev not found",
            "steiner wont work",
            "Import error: Negex",
            "Using keyword matching instead",
            "Import error: Funcassociate",
            "Make sure that funcassociate is in toolbox!",
            "DIAMOnD not found"
        ]
        # 添加一个标志来跟踪是否是首次输出
        self.is_first_output = True
        # 添加一个缓冲区来收集完整的JSON输出
        self.buffer_text = ""
        
    def readline(self):
        # 等待用户输入
        input_ready.clear()
        socketio.emit('waiting_input')
        input_ready.wait()
        return input_queue.get() + '\n'

    def write(self, text):
        # 处理bytes类型的输入
        if isinstance(text, bytes):
            text = text.decode(self.encoding)
            
        # 检查是否需要过滤掉这条消息
        if text.strip():
            # 如果是首次输出，跳过所有初始化消息
            if self.is_first_output and any(x in text for x in [
                "wrappers.py", "GenRev", "steiner", "Negex", 
                "Funcassociate", "DIAMOnD"
            ]):
                return
                
            should_filter = any(filter_text in text for filter_text in self.filter_texts)
            if not should_filter:
                self.is_first_output = False
                # 将文本添加到缓冲区
                self.buffer_text += text
                
                # 检查是否是一个完整的JSON输出
                if text.strip().endswith('}'):
                    # 发送完整的缓冲区内容
                    socketio.emit('output', {'data': self.buffer_text})
                    # 清空缓冲区
                    self.buffer_text = ""
                # 如果不是JSON输出，直接发送
                elif not any(json_marker in text for json_marker in ['"analysis_report":', '"message":']):
                    socketio.emit('output', {'data': text})
    
    # 添加必要的方法
    def flush(self):
        pass
    
    def isatty(self):
        return False
    
    def readable(self):
        return True
    
    def writable(self):
        return True
    
    def seekable(self):
        return False

    # 添加buffer属性
    @property
    def buffer(self):
        return self

# 路由设置
@app.route('/')
def index():
    return render_template('chat.html')

@socketio.on('user_input')
def handle_input(data):
    input_text = data['data']
    input_queue.put(input_text)
    input_ready.set()

def run_model():
    # 重定向标准输入输出
    web_io = WebIO()
    sys.stdin = web_io
    sys.stdout = web_io
    
    try:
        model_12_final.main()
    except Exception as e:
        socketio.emit('output', {'data': f'错误: {str(e)}'})

if __name__ == '__main__':
    # 在单独的线程中运行模型
    model_thread = threading.Thread(target=run_model)
    model_thread.daemon = True
    model_thread.start()
    
    # 启动Flask应用
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True) 