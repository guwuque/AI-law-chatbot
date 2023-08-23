#该应用创建工具共包含三个区域，顶部工具栏，左侧代码区，右侧交互效果区，其中右侧交互效果是通过左侧代码生成的，存在对照关系。
#顶部工具栏：运行、保存、新开浏览器打开、实时预览开关，针对运行和在浏览器打开选项进行重要说明：
#[运行]：交互效果并非实时更新，代码变更后，需点击运行按钮获得最新交互效果。
#[在浏览器打开]：新建页面查看交互效果。
#以下为应用创建工具的示例代码
import os
import gradio as gr
import pandas as pd
import requests
import json
import re
import sqlite3
import _thread as thread
import base64
import datetime
import hashlib
import hmac
import websocket
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

APPID = "1f6777d5"
APISecret = "MWVkZTA0NDI2NDU5NGE5ZGU5MGQ2MDMw"
APIKey = "97759935e91662a9b8bbb1f2041ba8ae"
sbase_url = 'ws://spark-api.xf-yun.com/v1.1/chat'
my_output = ""
click = 1
messages = []
con = sqlite3.connect('db.sqlite3', check_same_thread=False)
df_point = pd.read_sql_query('select * from lawpoint_lawpoint;', con)
law_name_list = df_point['law_name'].unique().tolist()
con.close()

class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, gpt_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(gpt_url).netloc
        self.path = urlparse(gpt_url).path
        self.gpt_url = gpt_url

    # 生成url
    def create_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成url
        url = self.gpt_url + '?' + urlencode(v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url


# 收到websocket错误的处理
def on_error(ws, error):
    print("### error:", error)


# 收到websocket关闭的处理
def on_close(ws,close_status_code, close_msg):
    print("### closed ###")


# 收到websocket连接建立的处理
def on_open(ws):
    thread.start_new_thread(run, (ws,))


def run(ws, *args):
    data = json.dumps(gen_params(appid=ws.appid, messages=ws.messages))
    ws.send(data)


# 收到websocket消息的处理
def on_message(ws, message):
    global my_output
    # print(message)
    data = json.loads(message)
    code = data['header']['code']
    if code != 0:
        print(f'请求错误: {code}, {data}')
        ws.close()
    else:
        choices = data["payload"]["choices"]
        status = choices["status"]
        content = choices["text"][0]["content"]
        print(content, end='')
        my_output= my_output + content
        if status == 2:
            ws.close()


def gen_params(appid, messages):
    """
    通过appid和用户的提问来生成请参数
    """
    data = {
        "header": {
            "app_id": appid,
            "uid": "1234"
        },
        "parameter": {
            "chat": {
                "domain": "general",
                "random_threshold": 0.5,
                "max_tokens": 2048,
                "auditing": "default"
            }
        },
        "payload": {
            "message": {
                "text": messages
            }
        }
    }
    return data


def sparkchat(messages):
    global my_output
    my_output = ""
    wsParam = Ws_Param(APPID, APIKey, APISecret,"ws://spark-api.xf-yun.com/v1.1/chat")
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
    ws.appid = APPID
    ws.messages = messages
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})


def get_from_sql(input_sql):
    conn = sqlite3.connect('db.sqlite3')
    c = conn.cursor()
    return_data = c.execute(input_sql).fetchone()[0]
    c.close()
    conn.close()
    return return_data


def generate(prompt_template, learn):
    try:
        law_name = get_from_sql("select law_name from lawpoint_lawpoint where law_name='" + learn + "' order by ('*')")
    except Exception as e:
        law_name = get_from_sql("select law_name from lawpoint_lawpoint order by ('*')")
    law_point = get_from_sql("select law_point from lawpoint_lawpoint where law_name='" + law_name + "' order by ('*')")
    role = get_from_sql("select prompt_template from prompt_prompt_manage where prompt_type='情境角色' order by ('*')")
    prompt_template = prompt_template.replace("{法律名称}", law_name)
    prompt_template = prompt_template.replace("{法律考点}", law_point)
    prompt_template = prompt_template.replace("{情境角色}", role)
    return prompt_template


def clear_txt(outputs):
    global messages, click
    click = 1
    messages = []
    continue_prompt_txt = "继续"
    return gr.Text.update(value=None)


def chatbot(input):
    global messages, click, my_output
    if input:
        if click == 1:
            prompt_template = get_from_sql(
                "select prompt_template from prompt_prompt_manage where prompt_type='法律情境' order by ('*')")
            prompt_template = generate(prompt_template, input)
        else:
            prompt_template = input
        print(prompt_template)
        messages.append({"role": "user", "content": prompt_template})
        sparkchat(messages)
        messages.append({"role": "assistant", "content": my_output})
        click = click + 1
        print(my_output)
        return my_output


def chatbot2(input):
    global messages, click, my_output
    if input:
        prompt_template = input
        messages.append({"role": "user", "content": prompt_template})
        sparkchat(messages)
        messages.append({"role": "assistant", "content": my_output})
        return my_output


def respond(message, chat_history):
    bot_ori_message = chatbot(message)
    bot_message = add_newline_before_abcd(bot_ori_message)
    chat_history.append((message, bot_message))
    if message != "继续":
        message = ""
    return message, chat_history


def add_newline_before_abcd(text):
    text = str(text)
    pattern = r'([A-D]\.)'
    result = re.sub(pattern, r'<br>\1', text)
    return result


def reload_point(radio_btn):
    global df_point
    law_point_list = df_point.loc[df_point['law_name'] == radio_btn, 'law_point'].unique().tolist()
    radio_return = gr.Radio.update(choices=law_point_list, value=None)
    return radio_return


def learn(law, point, chat_history):
    global messages, click
    click = 1
    messages = []
    prompt = "我是一名准备参加中国法考的学生，请为我提供" + law + "方向" + point + "考点的法律专业讲解和真题解析。"
    bot_ori_message = chatbot2(prompt)
    bot_message = add_newline_before_abcd(bot_ori_message)
    chat_history.append((prompt, bot_message))
    return chat_history



def main():
    global messages
    with gr.Blocks() as main:
        with gr.Row():
            with gr.Column():
                input_law_radio = gr.Radio(choices=law_name_list, label='请选择感兴趣的法律方向，点提交后生成情境')
                input_prompt_txt = gr.Textbox(lines=2, label="可以填写相关法考考点：如刑法、民法、行政法等")

                def update_textbox(choice):
                    input_law_radio.value = None
                    return choice

                input_law_radio.change(fn=update_textbox, inputs=input_law_radio, outputs=input_prompt_txt)
                continue_prompt_txt = gr.Textbox(lines=1, value="继续", visible=False)
                with gr.Row():
                    clear_btn = gr.Button(value="重新开始")
                    continue_btn = gr.Button(value="继续")
                    submit_btn = gr.Button(value="提交")
                with gr.Row():
                    input_law_radio2 = gr.Radio(choices=law_name_list, label='请选择想要进一步学习的法律方向,点击“我要学习”按钮生成学习资料')
                with gr.Row():
                    input_point_radio = gr.Radio(choices=[], label='法考考点')
                    input_law_radio2.change(reload_point, inputs=[input_law_radio2], outputs=[input_point_radio])
                with gr.Row():
                    clear_btn2 = gr.Button(value="清空学习内容")
                    submit_btn2 = gr.Button(value="我要学习")
            with gr.Column():
                outputs = gr.Chatbot(value=[], elem_id="chatbot_id", height=650)
        continue_btn.click(respond, [continue_prompt_txt, outputs], [continue_prompt_txt, outputs])
        clear_btn.click(clear_txt, inputs=[outputs], outputs=[outputs])
        clear_btn2.click(clear_txt, inputs=[outputs], outputs=[outputs])
        submit_btn.click(respond, [input_prompt_txt, outputs], [input_prompt_txt, outputs])
        submit_btn2.click(learn, [input_law_radio2, input_point_radio, outputs], [outputs])
    main.launch()

main()