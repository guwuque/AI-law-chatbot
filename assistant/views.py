# importing render and redirect
from django.shortcuts import render, redirect
# importing the openai API
import openai
# import the generated API key from the secret_key file
from .secret_key import API_KEY
# loading the API key from the secret_key file
openai.api_key = API_KEY


# this is the home view for handling home page logic
def home(request):
    try:
        # if the session does not have a messages key, create one
        if 'messages' not in request.session:
            request.session['messages'] = [
                {"role": "system", "content": "输入想学习的实体法、程序法，让我们开始吧"},
            ]
        if request.method == 'POST':
            # get the prompt from the form
            learn = request.POST.get('prompt')
            if not learn.encode().isalpha():
                prompt = "我是一个法律专业相关的学生，想系统学习学习" + learn + "。请设计一个" + learn + "案件的情境化场景，由我扮演律师，你负责扮演我的委托代理人、法院的法官等角色，并根据情境依次提出5道有4个选项的选择题，包括" + learn + "总则、分则以及" + learn + "诉讼相关知识点，类似文字冒险游戏，每个题目出现后，都暂停一下情境让我可以立刻通过会话去选择答案进行回答，和你产生交互并通过我的回答一步一步的推动故事情境的发展，最后根据回答的结果的对错，对我这个律师进行三个等级的评价，现在情境开始！."
            else:
                prompt = learn
            # get the temperature from the form
            temperature = float(request.POST.get('temperature', 0.1))
            # append the prompt to the messages list
            request.session['messages'].append({"role": "user", "content": prompt})
            # set the session as modified
            request.session.modified = True
            # call the openai API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=request.session['messages'],
                temperature=temperature,
                max_tokens=1000,
            )
            # format the response
            formatted_response = response['choices'][0]['message']['content']
            # append the response to the messages list
            request.session['messages'].append({"role": "assistant", "content": formatted_response})
            request.session.modified = True
            # redirect to the home page
            context = {
                'messages': request.session['messages'],
                'prompt': '',
                'temperature': temperature,
            }
            return render(request, 'assistant/home.html', context)
        else:
            # if the request is not a POST request, render the home page
            context = {
                'messages': request.session['messages'],
                'prompt': '',
                'temperature': 0.8,
            }
            return render(request, 'assistant/home.html', context)
    except Exception as e:
        print(e)
        # if there is an error, redirect to the error handler
        return redirect('error_handler')


def new_chat(request):
    # clear the messages list
    request.session.pop('messages', None)
    if request.method == 'POST':
        prompt = request.POST.get('prompt_template')
    return redirect('home')


# this is the view for handling errors
def error_handler(request):
    return render(request, 'assistant/404.html')
