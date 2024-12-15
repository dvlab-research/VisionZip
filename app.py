import subprocess
import gradio as gr
import time

def start_controller():
    subprocess.Popen(['python', '-m', 'llava.serve.controller', '--host', '0.0.0.0', '--port', '10000'])
    time.sleep(3)  


def start_gradio_web_server():
    subprocess.Popen(['python', '-m', 'llava.serve.gradio_web_server', '--controller', 'http://localhost:10000', '--model-list-mode', 'reload'])
    time.sleep(3)  

def start_model_worker():
    subprocess.Popen(['python', '-m', 'llava.serve.model_worker', '--host', '0.0.0.0', '--controller', 'http://localhost:10000', '--port', '40000', '--worker', 'http://localhost:40000', '--model-path', '/root/MODELS/llava-v1.5-7b'])

def gradio_interface():
    gr.Interface(fn=lambda x: x, inputs="text", outputs="text").launch()


if __name__ == "__main__":
    start_controller()
    start_gradio_web_server()
    start_model_worker()
    gradio_interface()
