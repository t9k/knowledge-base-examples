import gradio as gr
import time

def response(message, history):
    # 假定 LLM 分段输出如下几部分
    chunks = [
        "这是第一段输出……",
        "这是第二段输出……",
        "这是第三段输出，接近尾声。",
        "这是最后一段输出：结束！"
    ]
    resp = ""
    for chunk in chunks:
        time.sleep(0.7)  # 模拟生成延迟
        resp += chunk
        yield resp

demo = gr.ChatInterface(
    fn=response,
    type="messages",
    save_history=True,
)
demo.launch()
