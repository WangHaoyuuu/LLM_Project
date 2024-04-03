## 第二章调用大模型的API
1. 基本概念
    1. Prompt:访问大模型的输入，用来指导模型生成所需的输出
    2. Completion:大模型返回的结果
    3. Temperature:LLM生成结果的随机性
    4. System Prompt:在整个会话过程中持久地影响模型的回复,是一种初始化设定
2. 调用Chatgpt
    1. 获取API后为什么需要保存到.env文件中
        - 保护API密钥，简化配置，提高安全性
        - 以这种形式保存 OPENAI_API_KEY="sk-..."
        ```
        import os
        import openai
        from dotenv import load_dotenv, find_dotenv
        # find_dotenv()
        # load_dotenv()
        _ = load_dotenv(find_dotenv())
        # _ 通常作为一个临时变量，不关心该变量
        openai.api_key = os.environ['OPENAI_API_KEY']
        ```
    2. 调用OpenAI原生接口
        - 需要用到ChatCompletion API
            ```
            import openai
            completion = openai.ChatCompletion.create
            (
                model="gpt-3.5-turbo",
                # message 是prompt
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."}
                    {"role": "user", "content": "Who won the world series in 2020?"}
                ]

            # 第一个role
            # 系统角色:系统角色扮演一个系统，可以用来设置对话的上下文，例如设定对话的背景、主题或规则

            # 第二个role
            # 用户角色:用户角色扮演一个用户，可以用来设置对话的输入，例如输入一个问题或指令
            )
            # 返回的completion 是一个字典，包含"choices"键，该键是一个列表，包含一个字典，该字典包含"message"键，该键是一个字典，包含"content"键，该键包含模型的回复
            # 打印模型的回复
            print(completion["choices"][0]["message"]["content"])
            ```
        - API 常用参数
            1. model: 模型名称，例如"gpt-3.5-turbo"
            2. messages: 对话内容，是一个列表，列表中的每个元素是一个字典，字典包含"role"和"content"和"assistan"三个键，分别表示角色、内容、助手
            3. temperature: 温度参数，用于控制回复的随机性，值越高，回复越随机
            4. max_tokens: 生成回复的最大令牌数
        - 对OpenAI的API进行封装
            - 封装一个函数，用于调用OpenAI的API(一般用不到system prompt和assistant prompt)
                ```
                    # 封装一个函数，用于调用OpenAI的API
                    def chat_with_gpt(prompt, model="gpt-3.5-turbo", temperature=0.6):
                        messages = [{"role": "user", "content": prompt}]
                        response = openai.ChatCompletion.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                        )
                        return
                        response["choices"][0]["message"]["content"]
                ```

    3. 基于LangChain调用ChatGPT
        - 导入OpenAI的对话模型
            ```
                from langchain.chat_models import ChatOpenAI
            ```
        - 实例化ChatOpenAI模型
            ```
                chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.6)
            ```
        - 使用Template来设置prompt
            ```
                from langchain.prompts import ChatPromptTemplate
                template_string = """
                Translate the following English text to Chinese:
                text: """{text}"""
                """
                # 实例化ChatPromptTemplate
                chat_template = ChatPromptTemplate.from_template(template_string)

                # 后续使用时，改变text的值即可

                text = "Hello, how are you?"
                # 调用chat.generate方法生成回复
                message = chat_template.format_prompt(text=text)
                print(message)

                # 使用实例化的类型传入prompt
                response = chat(message)
                print(response)
            ```
3. 调用百度文心
    1. 调用百度文心的原生接口
        - 百度文心需要用到API_Key和Secret_Key，基于这两个Key获取access_token
            ```
                import requests
                import json
                def get_access_token(api_key, secret_key):
                    # 设置请求的URL
                    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
                    # 设置POST请求
                    payload = json.dumps("")
                    # 将请求体（payload）转换为JSON格式。
                    headers = {
                        'Content-Type': 'application/json',
                        'Accept':
                        'application/json'
                        # 是一个 MIME 类型，表示 JSON 格式的数据。
                    }
                    # 发送请求
                    response = requests.request("POST", url, headers=headers, data=payload)
                    # 获取access_token
                    access_token = response.json().get("access_token")
                    return access_token
                    # 后续使用 access_token 即可调用百度文心大模型。
            ```
        - 调用百度文心的原生接口
            ```
            def get_wenxin(prompt):
                # 设置请求的URL
                url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token={access_token}"
                # 配置POST参数
                payload = json.dumps({
                    "message":[
                        {
                            "role":"user",
                            "content":"{}".format(prompt)
                    }]
                })
                headers = {
                    'Content-Type': 'application/json'
                }
                # 发送请求
                response = requests.request("POST", url, headers=headers, data=payload)
                # 获取回复
                reply = json.loads(response.text)
                print(reply["result"])
                
            get_wenxin("你好")
            ```
        - 同样封装一个函数后使用
            ```
            def chat_with_wenxin(prompt, temperature=0.6, access_token="your_access_token"):
            '''
            prompt: 提示词
            temperature: 温度参数
            access_token: 百度文心的access_token
            '''
            url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token={access_token}"
            
            # 配置POST参数
            payload = json.dumps({
                "message":[
                    {
                        "role":"user",
                        "content":"{}".format(prompt)
                    }
                ],
                "temperature":temperature
            })
            headers = {
                'Content-Type': 'application/json'
            })
            # 发送请求
            response = requests.request("POST", url, headers=headers, data=payload)
            # 获取回复
            reply = json.loads(response.text)
            return reply["result"]

            # 调用函数
            prompt = "你好"
            temperature = 0.6
            access_token = "your_access_token"
            chat_with_wenxin(prompt, temperature, access_token)
            ```
    2. 利用LangChain调用百度文心
        - 原生的 LangChain 是不支持文心调用的, 我们需要自定义支持调用文心的LLM
    
