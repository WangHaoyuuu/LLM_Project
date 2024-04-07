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
            ```
            from wenxin_llm import Wenxin_LLM
            ```
        - 将文心的api_key和secret_key存储在.env文件中,并使用以下代码加载
            ```
            from dotenv import load_dotenv, find_dotenv
            import os
            _ = load_dotenv(find_dotenv())
            # 获取API_KEY和SECRET_KEY
            wenxin_api_key = os.environ["WENXIN_API_KEY"]
            wenxin_secret_key = os.environ["WENXIN_SECRET_KEY"]
            ```
        - 实例化Wenxin_LLM
            ```
                wenxin_llm = Wenxin_LLM(wenxin_api_key, wenxin_secret_key)
                wenxin_llm("你好")
            ```
4. 调用讯飞星火
    1. 星火 API 需要使用 WebSocket 来进行调用
    2. 调用原生星火API
        - 对test.py中的调用逻辑进行讲解
        1. 配置密匙信息
            ```
            import SparkApi

            # 设置密匙信息
            appid = ""  
            api_secret = ""
            api_key = ""

            # 用于配置大模型版本
            domain = "general" # v1.5版本

            # 云端环境的服务地址
            Spark_url = "ws://spark-api.xf-yun.com/v1.1/chat" # v1.5版本 
            ```

        2. 星火的调用传参(列表中包括role和Prompt)
            ```
            def getText(role, content, text = []):
                # role 表示说话的角色，content 表示Prompt的内容
                jsoncon = {}
                jsoncon["role"] = role
                jsoncon["content"] = content
                text.append(jsoncon)
                return text
            
            questions = getText("user", "你好")            
            ```
        3. 调用SparkApi.py中的main函数
            ```
            response = SparkApi.main(appid, api_key, api_secret, Spark_url, domain, questions)
            response
            ```
        
        4. 同一API调用方式
            - 由于星火使用的是WebSocket，因此不能使用requests进行访问，通过FastAPI将星火API封装成本地的API(spark_api.py)文件在chapter2文件夹
                ```
                    # 使用uvicorn命令启动
                    uvicorn spark_api:app
                ```
            - 启动后会在8000端口开启api服务
            - 调用方式，发起Request请求
                ```
                import requests
                api_url = "http://127.0.0.1:8000/spark"
                headers = {
                    "Content-Type": "application/json"}
                data = {
                    "prompt": "你好",
                    "temperature": 0.6
                    "max_tokens": 3096
                }

                response = requests.post(api_url, headers=headers, json=data)
                response.text
                ```

            - 同样的封装一个函数后使用
                ```
                def chat_with_spark(prompt,temperature=0.6, max_tokens=3096):
                    api_url = "http://127.0.0.1:8000/spark"
                    headers = {
                        "Content-Type": "application/json"
                    }
                    data = {
                        "prompt": prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                    response = requests.post(api_url, headers=headers, json=data)
                    return response.text

                chat_with_spark("你好")
                '''
5. 调用ChatGLM
    1. 智谱提供了SDK和原生HTTP来实现调用，建议使用SDK来获得更好的编程体验
        ```
        pip install zhipuai
        ```
    2. 调用SDK
        ```
        import zhipuai
        zhipuai.api_key = "your_api_key"
        model = "chatglm_std"
        ```
    3. 智谱的调用需要传入列表
        ```
        def getText(role, content, text = []):
            # role 表示说话的角色，content 表示Prompt的内容
            jsoncon = {}
            jsoncon["role"] = role
            jsoncon["content"] = content
            text.append(jsoncon)
            return text
        
        questions = getText("user", "你好")
        questions

    4. 调用智谱SDK中的invoke函数
        ```
            response = zhipuai.model_api.invoke(
                model = model, 
                prompt = questions,)
            response
        ```
    5. 使用LangChain调用智谱
        ```
        from zhipuai_llm import ZhipuAILLM
        zhipuai_llm = ZhipuAILLM(model = "chatglm_std", temperature=0, zhipuai_api_key=zhipuai.api_key)
        zhipuai_llm("你好")
        ```
6. 调用智谱AI生成embedding
    ```
    import zhipuai
    zhipuai.api_key = "your_api_key"
    model = "text_embedding"
    ```
    - 自定义要生成embedding的文本
        ```
            text = "要生成embeding的输入文本"
        ```
    - 调用智谱SDK中的invoke函数
        ```
            response = zhipuai.model_api.invoke(
                model = model,
                prompt = text,
            )
        ```
    - 返回是字典的格式，查看code是否为200判断请求是否成功
        ```
            print(response['code'])
        ```
    - 返回的embedding和token被存放在data中，可以查看embeding
    的长度
        ```
            print(len(response['data']['embeding']))
        ```
    - 查看输入的token数量
        ```
            print(response['data']['usage']['prompt_tokens'])
    - 使用LangChain调用智谱AI生成embedding
        ```
            import zhipuai
            from zhipuai_embedding import ZhipuAIEmbedding
            zhipuai.api_key = "your_api_key"
            zhipuai_embedding = ZhipuAIEmbedding(zhipuai_api_key=zhipuai.api_key)
            # 可以生成query的embedding
            # query的embedding可以用于搜索
            query_embedding = zhipuai_embedding.embed_query("要生成embeding的输入文本")
            # 也可以生成doc_list的embedding
            # doc_list的embedding可以用于生成摘要
            doc_list_embedding = zhipuai_embedding.embed_documents("要生成embeding的输入文本")
        ```
7. Langchain组件详解
    1. 模型的输入输出
        - 输入：管理prompt
        - 输出：调用API获取回复
        - 主要包含Prompts, Language Models, Output Parsers
    2. 数据连接
        - 大语言模型的知识来源于其训练数据集，并没有用户信息
        - 加载：Document loaders
        - 转存：Document transformers，Text embedding models，Vector stores
        - 查询数据：Retrievers
    3. Chain(链)
        - 独立使用大模型能应对简单的任务，更复杂的需求，需要将多个大模型链式组合，或与其他组件进行链式调用
        - LLMChain是一个简单而强大的链，许多链以它为基础
            ```
            from langchain.chat_models import ChatOpenAI
            from langchain.prompts import ChatPromptTemplate
            from langchain.chain import LLMChain
            
            # 实例化大模型
            llm = ChatOpenAI(temperature=0.6)
            # 实例化prompt
            prompt = ChatPromptTemplate.from_messages([
            "描述制造{product}的一个公司的最佳名称是什么?"
            ])
            # 将LLM和prompt组合成LLMChain
            chain = LLMChain(llm=llm, prompt=prompt)
            # 调用LLMChain
            product = "手机"
            chain.run(product)
            ```
        - 除了LLMChain，还有
            - RouterChain:根据输入的内容，选择不同的LLM进行调用
            - Transformation:将输入的内容进行转换，再调用LLM
            - SimpleSequentialChain:最简单的序列链形式，上一个步骤的输出是下一个步骤的输入
            - sequence_chains:简单顺序链的更复杂形式，允许多个输入/输出
            ```
            # SimpleSequentialChain 的代码示例
            from langchain.chains import SimpleSequentialChain
            llm = ChatOpenAI(temperature=0.6)

            # 创建第一个LLMChain
            prompt1 = ChatPromptTemplate.from_messages([
                "描述制造{product}的一个公司的最佳名称是什么?"
            ])
            chain1 = LLMChain(llm=llm, prompt=prompt1)

            # 创建第二个LLMChain
            prompt2 = ChatPromptTemplate.from_messages([
                "写一个20字的描述对于{product}公司"
            ])
            chain2 = LLMChain(llm=llm, prompt=prompt2)

            # 创建SimpleSequentialChain
            overall_chain = SimpleSequentialChain(chains=[chain1, chain2]，verbose=True)

            product = "手机"
            overall_chain.run(product)
            ```
    4. 记忆(Memory)
        - 在LangChain中，记忆指的是LLM的短期记忆
    5. 智能体(Agents)
        - 代理作为语言模型的外部模块，可提供计算、逻辑、检索等功能的支持，使语言模型获得异常强大的推理和获取信息的超能力
    6. 回调(Callback)
        - 能连接到LLM的各个阶段，充当日志功能
            - CallbackHandler：记录日志
            - CallbackManager：封装管理所有的CallbackHandler
        - 在哪里传入回调
            1. **构造函数回调**构造函数LLMChain(callbacks=[handler], tags=['a-tag']) ，它将被用于对该对象的所有调用，并且将只针对该对象
            2. **请求回调**在用于发出请求的 call() / run() / apply() 方法中chain.call(inputs, callbacks=[handler]) ，它将仅用于该特定请求，以及它包含的所有子请求
            - verbose 参数在整个 API 的大多数对象（链、模型、工具、代理等）上都可以作为构造参数使用
            - 构造函数回调对诸如日志、监控等用例最有用
            - 请求回调对流媒体等用例最有用
8. 完成
