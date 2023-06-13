# AI-law-chatbot
项目基于Django-admin和simpleui，用于简单管理数据存储。实现法律情景化AI学习。
## 1、请自行在根目录下建立static目录，用于存储simpleui的静态缓存文件。
python manage.py collectstatic

## 2、gardio的界面
gen_chat_multi.py用于启动查询界面，可以稍微修改一下代码，支持对本地简单文件的知识结合，就是比较费openai的token。

## 3、混合本地知识库查询
可以建立locallaw目录，自行拉取langchain-chatglm的项目来结合本地知识
