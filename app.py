import gradio as gr
from inference import run

history = []

def chat(user_msg, chat_history):
    global history
    if not user_msg.strip():
        return "", chat_history
    response = run(user_msg, history)
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": response})
    chat_history = chat_history + [[user_msg, response]]
    return "", chat_history

def clear_chat():
    global history
    history = []
    return []

with gr.Blocks(title="Pocket Agent") as demo:
    gr.Markdown("# 🤖 Pocket Agent")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="e.g. What's the weather in Karachi in Celsius?")
    with gr.Row():
        submit = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear")
    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    submit.click(chat, [msg, chatbot], [msg, chatbot])
    clear.click(clear_chat, outputs=[chatbot])

demo.launch()