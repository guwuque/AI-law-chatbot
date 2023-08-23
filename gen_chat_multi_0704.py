import os
import nltk
import torch
from chatglm_llm import ChatGLM
from langchain.chat_models import ChatOpenAI    # 调AI接口的
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from modelscope_hub import ModelScopeEmbeddings
import openai
import gradio as gr
import sqlite3
import pandas as pd
from datetime import datetime

nltk.data.path.append('./nltk_data')

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

embedding_model_dict = {
    "corom-tiny": "damo/nlp_corom_sentence-embedding_chinese-tiny",
    "corom-tiny-ecom": "damo/nlp_corom_sentence-embedding_chinese-tiny-ecom",
    "corom-base-ecom": "damo/nlp_corom_sentence-embedding_chinese-base-ecom",
    "corom-base": "damo/nlp_corom_sentence-embedding_chinese-base",
}

llm_dict = {
    'ChatGLM-6B': {
        'model_name': 'ZhipuAI/ChatGLM-6B',
        'model_revision': 'v1.0.15',
    },
    'ChatGLM-6B-int8': {
        'model_name': 'thomas/ChatGLM-6B-Int8',
        'model_revision': 'v1.0.3',
    },
    'ChatGLM-6B-int4': {
        'model_name': 'ZhipuAI/ChatGLM-6B-Int4',
        'model_revision': 'v1.0.3',
    }
}


def aichatbot(input):
    if input:
        messages.append({"role": "user", "content": input})
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return reply


def reload_point(radio_btn):
    law_point_list = df_point.loc[df_point['law_name'] == radio_btn, 'law_point'].unique().tolist()
    radio_return = gr.Radio.update(choices=law_point_list, value=None)
    return radio_return


def reload_action(radio_btn):
    return gr.Text.update(value=radio_btn)


def clear_txt(input_txt):
    return gr.Text.update(value='')


def gen_prompt(input_txt, law_radio, point_radio):
    try:
        return gr.Text.update(value=input_txt.format(law_radio, point_radio))
    except:
        return gr.Text.update(value='Some Error')


def direct_prompt(radio_btn):
    return gr.Text.update(value=radio_btn)


def save_chat():
    try:
        t_df = pd.DataFrame(data=None, columns=['chat_datetime', 'chat_content'])
        t_df.loc[1] = [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(messages)]
        t_df.to_sql('message_chatmsg', con, if_exists="append", index=False)
        return gr.Radio.update(value='Save Ok')
    except:
        return gr.Radio.update(value='Save Failed')


# -----------langchain-chatglm-webui-----------
def clear_session():
    return '', None

def init_knowledge_vector_store(embedding_model, filepath):

    embeddings = ModelScopeEmbeddings(
        model_name=embedding_model_dict[embedding_model], )

    loader = UnstructuredFileLoader(filepath, mode="elements")
    docs = loader.load()

    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def get_knowledge_based_answer(
    query,
    large_language_model,
    vector_store,
    VECTOR_SEARCH_TOP_K,
    web_content,
    chat_history=[],
    history_len=3,
    temperature=0.01,
    top_p=0.9,
):
    if web_content:
        prompt_template = f"""基于以下已知信息，简洁和专业的来回答用户的问题。
                            如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
                            已知网络检索内容：{web_content}""" + """
                            已知内容:
                            {context}
                            问题:
                            {question}"""
    else:
        prompt_template = """基于以下已知信息，请简洁并专业地回答用户的问题。
            如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。不允许在答案中添加编造成分。另外，答案请使用中文。

            已知内容:
            {context}

            问题:
            {question}"""
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    chatLLM = ChatGLM()
    chatLLM.model_name = llm_dict[large_language_model]['model_name']
    chatLLM.model_revision = llm_dict[large_language_model]['model_revision']

    chatLLM.history = chat_history[-history_len:] if history_len > 0 else []
    chatLLM.temperature = temperature
    chatLLM.top_p = top_p

    knowledge_chain = RetrievalQA.from_llm(
        llm=chatLLM,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": VECTOR_SEARCH_TOP_K}),
        prompt=prompt)
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}")

    knowledge_chain.return_source_documents = True
    result = knowledge_chain({"query": query})

    return result['result']

def predict(input,
            large_language_model,
            embedding_model,
            file_obj,
            VECTOR_SEARCH_TOP_K,
            history_len,
            temperature,
            top_p,
            # use_web,
            history=None):
    if history == None:
        history = []
    print(file_obj.name)
    # if use_web == 'True':
    #     web_content = search_web(query=input)
    # else:
    #     web_content = ''
    vector_store = init_knowledge_vector_store(embedding_model, file_obj.name)

    resp = get_knowledge_based_answer(
        query=input,
        large_language_model=large_language_model,
        vector_store=vector_store,
        VECTOR_SEARCH_TOP_K=VECTOR_SEARCH_TOP_K,
        web_content=web_content,
        chat_history=history,
        history_len=history_len,
        temperature=temperature,
        top_p=top_p,
    )
    print(resp)
    history.append((input, resp))
    return '', history, history


def main():
    # 创建法律考点
    with gr.Blocks() as lawpoint:
        with gr.Row():
            with gr.Column():
                input_law_radio = gr.Radio(choices=law_name_list, label='法律方向')
                input_point_radio = gr.Radio(choices=[], label='重要考点')
                input_action_radio = gr.Radio(choices=prompt_templates, label='学习模式')
                input_prompt_txt = gr.Textbox(default=prompt_templates[0], lines=7, label="试试ChatGPT")
                with gr.Row():
                    clear_btn = gr.Button(value="清除内容")
                    generate_btn = gr.Button(value="生成提示词")
                    submit_btn = gr.Button(value="提交问题")
                    save_btn = gr.Button(value="对话存储")
            with gr.Column():
                save_status = gr.Radio(choices=['Save Ok', 'Save Failed'], label='对话存储情况', interactive=False)
                outputs = gr.outputs.Textbox(label="ChatGPT回复：")
        input_law_radio.change(reload_point, inputs=[input_law_radio], outputs=[input_point_radio])
        input_action_radio.change(reload_action, inputs=[input_action_radio], outputs=[input_prompt_txt])
        clear_btn.click(clear_txt, inputs=[input_prompt_txt], outputs=[input_prompt_txt])
        generate_btn.click(gen_prompt, inputs=[input_prompt_txt, input_law_radio, input_point_radio],
                           outputs=[input_prompt_txt])
        submit_btn.click(aichatbot, inputs=[input_prompt_txt], outputs=[outputs])
        save_btn.click(save_chat, inputs=[], outputs=[save_status])
    # 创建模拟法庭
    with gr.Blocks() as lawcourt:
        with gr.Row():
            with gr.Column():
                input_law_radio = gr.Radio(choices=prompt_list, label='提示词模板')
                input_prompt_txt = gr.Textbox(default=prompt_templates[0], lines=7, label="试试ChatGPT")
                with gr.Row():
                    clear_btn = gr.Button(value="清除内容")
                    submit_btn = gr.Button(value="提交问题")
                    save_btn = gr.Button(value="对话存储")
            with gr.Column():
                save_status = gr.Radio(choices=['Save Ok', 'Save Failed'], label='对话存储情况', interactive=False)
                outputs = gr.Textbox(label="ChatGPT回复：")
        input_law_radio.change(direct_prompt, inputs=[input_law_radio], outputs=[input_prompt_txt])
        clear_btn.click(clear_txt, inputs=[input_prompt_txt], outputs=[input_prompt_txt])
        submit_btn.click(aichatbot, inputs=[input_prompt_txt], outputs=[outputs])
        save_btn.click(save_chat, inputs=[], outputs=[save_status])
    # local_index = construct_index("docs")
    with gr.Blocks() as local_law:
        gr.Markdown("""<h1><center>LangChain-ChatLLM-Webui</center></h1>
        <center><font size=3>
        本部分基于LangChain+chatGLMS-webui项目, 提供基于本地知识的自动问答应用. <br>
        </center></font>
        """)
        with gr.Row():
            with gr.Column(scale=1):
                model_choose = gr.Accordion("模型选择")
                with model_choose:
                    large_language_model = gr.Dropdown(
                        ["ChatGLM-6B", "ChatGLM-6B-int4", 'ChatGLM-6B-int8'],
                        label="large language model",
                        value="ChatGLM-6B-int8")

                    embedding_model = gr.Dropdown(list(
                        embedding_model_dict.keys()),
                        label="Embedding model",
                        value="corom-tiny")

                file = gr.File(label='请上传知识库文件',
                               file_types=['.txt', '.md', '.docx'])
                # use_web = gr.Radio(["True", "False"],
                #                    label="Web Search",
                #                    value="False")
                model_argument = gr.Accordion("模型参数配置")

                with model_argument:

                    VECTOR_SEARCH_TOP_K = gr.Slider(
                        1,
                        10,
                        value=6,
                        step=1,
                        label="vector search top k",
                        interactive=True)

                    HISTORY_LEN = gr.Slider(0,
                                            3,
                                            value=0,
                                            step=1,
                                            label="history len",
                                            interactive=True)

                    temperature = gr.Slider(0,
                                            1,
                                            value=0.01,
                                            step=0.01,
                                            label="temperature",
                                            interactive=True)
                    top_p = gr.Slider(0,
                                      1,
                                      value=0.9,
                                      step=0.1,
                                      label="top_p",
                                      interactive=True)

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label='ChatLLM').style(height=400)
                message = gr.Textbox(label='请输入问题')
                state = gr.State()

                with gr.Row():
                    clear_history = gr.Button("🧹 清除历史对话")
                    send = gr.Button("🚀 发送")

                    send.click(predict,
                               inputs=[
                                   message, large_language_model,
                                   embedding_model, file, VECTOR_SEARCH_TOP_K,
                                   HISTORY_LEN, temperature, top_p, # use_web,
                                   state
                               ],
                               outputs=[message, chatbot, state])
                    clear_history.click(fn=clear_session,
                                        inputs=[],
                                        outputs=[chatbot, state],
                                        queue=False)

                    message.submit(predict,
                                   inputs=[
                                       message, large_language_model,
                                       embedding_model, file,
                                       VECTOR_SEARCH_TOP_K, HISTORY_LEN,
                                       temperature, top_p, # use_web,
                                       state
                                   ],
                                   outputs=[message, chatbot, state])
        gr.Markdown("""提醒：<br>
        1. 更改LLM模型前请先刷新页面，否则将返回error（后续将完善此部分）. <br>
        2. 使用时请先上传自己的知识文件，并且文件中不含某些特殊字符，否则将返回error. <br>
        3. 请勿上传或输入敏感内容，否则输出内容将被平台拦截返回error.<br>
        4. 有任何使用问题，请通过[问题交流区](https://modelscope.cn/studios/thomas/ChatYuan-test/comment)或[Github Issue区](https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui/issues)进行反馈. <br>
        """)
    multi_gr = gr.TabbedInterface(([lawpoint, lawcourt, local_law]),
                                  ["法律考点", "模拟法庭", "本地知识库应用"])
    multi_gr.launch(server_name='0.0.0.0',server_port=8010)


if __name__ == "__main__":
    openai.api_key = "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"
    os.environ["OPENAI_API_KEY"] = openai.api_key
    con = sqlite3.connect('db.sqlite3', check_same_thread=False)
    df_point = pd.read_sql_query('select * from lawpoint_lawpoint;', con)
    law_name_list = df_point['law_name'].unique().tolist()
    df_prompt = pd.read_sql_query('select * from prompt_prompt_manage limit 5;', con)
    prompt_list = df_prompt['prompt_template'].tolist()
    prompt_templates = ["我是一名准备参加中国司法考试的学生。请为我提供[{0}]方向[{1}]考点的法律专业讲解和真题解析。",
                        "我是一名中国的实习律师。你作为一名专业律师，为我提供一个[{0}]方向[{1}]考点的法律场景，并对我的回答进行[好]或[差]的评价。",]
    messages = [{"role": "system", "content": "You are a helpful and kind AI Assistant."}, ]
    main()
    con.close()
