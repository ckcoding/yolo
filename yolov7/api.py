import os
import subprocess
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# 输入文件和输出目录
INPUT_VIDEO = "./fire.mp4"  # 本地已有视频文件路径
OUTPUT_FOLDER = "./output_hls"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def convert_video_to_m3u8(input_path, output_dir, segment_time=10):
    # 定义输出文件路径
    output_file = os.path.join(output_dir, 'output.m3u8')
    
    # 修改 FFmpeg 命令，添加视频编码和音频编码参数
    command = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',  # 使用 H.264 编码
        '-c:a', 'aac',      # 使用 AAC 音频编码
        '-hls_time', str(segment_time),
        '-hls_list_size', '0',
        '-f', 'hls',
        output_file
    ]
    
    # 执行命令并捕获输出
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        return output_file
    except subprocess.CalledProcessError as e:
        error_message = f"FFmpeg转换失败:\n命令: {' '.join(command)}\n错误输出: {e.stderr}"
        raise RuntimeError(error_message)

# 静态文件访问，允许访问 .m3u8 文件
@app.route('/output_hls/<path:filename>')
def serve_m3u8(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

# 定义转换视频的接口
@app.route('/convert', methods=['GET'])
def convert():
    try:
        output_path = convert_video_to_m3u8(INPUT_VIDEO, OUTPUT_FOLDER)
        return jsonify({
            "message": "视频转换成功",
            "output_m3u8_url": f"http://localhost:5000/output_hls/output.m3u8"
        }), 200
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 启动服务器时自动转换视频
    try:
        convert_video_to_m3u8(INPUT_VIDEO, OUTPUT_FOLDER)
        print("视频已成功转换为 m3u8 格式，可以通过 /output_hls/output.m3u8 访问。")
    except RuntimeError as e:
        print("视频转换失败:", e)

    app.run(debug=True, port=5000)
