import csv
import time

from transformers import AutoModel, AutoTokenizer
import streamlit as st
from streamlit_chat import message

st.set_page_config(
    page_title="ChatGLM-6b 演示",
    page_icon=":robot:"
)

filename = 'command_logs.csv'

csv_file = open(filename, mode='a', encoding='utf8')
fieldnames = ['time', 'user', 'query', 'response']
writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

# 如果文件是空文件，添加表头
if csv_file.tell() == 0:
    writer.writeheader()


@st.cache_resource
def get_model():
    chatglm_6b_path = "D:\huggingface\chatglm-6b"
    chatglm_6b_int4_path = "D:\huggingface\chatglm-6b-int4-qe"

    tokenizer = AutoTokenizer.from_pretrained(chatglm_6b_path, trust_remote_code=True)
    # model = AutoModel.from_pretrained(chatglm_6b_path, trust_remote_code=True).half().quantize(4).cuda()
    model = AutoModel.from_pretrained(chatglm_6b_int4_path, trust_remote_code=True).half().cuda()

    # tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    # model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    model = model.eval()
    return tokenizer, model


MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2


def predict_with_user(user_id, input, history=None):
    if not user_id:
        with container:
            st.write("请输入你的名称，这涉及到多轮对话的服务体验")
        return
    history_list = st.session_state["state"]
    history = history_list.get(user_id)
    tokenizer, model = get_model()
    if history is None:
        history = []

    with container:
        if len(history) > 0:
            for i, (query, response) in enumerate(history):
                message(query, avatar_style="big-smile", key=str(i) + "_user")
                message(response, avatar_style="bottts", key=str(i))

        message(input, avatar_style="big-smile", key=str(len(history)) + "_user")
        st.write("AI正在回复:")
        with st.empty():
            for response, history in model.stream_chat(tokenizer, input, history):
                query, response = history[-1]
                st.write(response)
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        writer.writerow({'time': current_time, 'user': user_id, 'query': history[-1][0], 'response': history[-1][1]})
        csv_file.flush()  # 确保数据被写入文件
    history_list.update({user_id: history})
    st.session_state["state"] = history_list


def predict(user_id, input, history=None):
    tokenizer, model = get_model()
    if history is None:
        history = []

    with container:
        if len(history) > 0:
            for i, (query, response) in enumerate(history):
                message(query, avatar_style="big-smile", key=str(i) + "_user")
                message(response, avatar_style="bottts", key=str(i))

        message(input, avatar_style="big-smile", key=str(len(history)) + "_user")
        st.write("AI正在回复:")
        with st.empty():
            for response, history in model.stream_chat(tokenizer, input, history):
                query, response = history[-1]
                st.write(response)

    return history


container = st.container()

# create a prompt text for the text generation
userID_text = st.text_area(label="你的名字",
                           height=20,
                           placeholder="请在这儿输入你的唯一ID，用于实现多轮对话")

prompt_text = st.text_area(label="用户命令输入",
                           height=100,
                           placeholder="请在这儿输入您的命令")

if 'state' not in st.session_state:
    st.session_state['state'] = {}

get_model()

if st.button("发送", key="predict"):
    with st.spinner("AI正在思考，请稍等........"):
        # text generation
        predict_with_user(userID_text, prompt_text, st.session_state["state"])
        # st.session_state["state"] = predict(userID_text, prompt_text, st.session_state["state"])
