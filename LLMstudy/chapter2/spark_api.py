"""
    将星火API封装成本地API
"""
from fastapi import FastAPI
from pydantic import BaseModel
import os
import SparkApiSelf

app = FastAPI()

# 定义一个数据模型，接收POST请求的数据

class Data(BaseModel):
    prompt: str
    max_tokens: int
    temperature: float
    # 是否多轮对话
    if_list: bool = False

# 定义一个构造参数的函数
def getText(role, content, text=[]):
    # role 为角色，content 为Prompt
    jsoncon = {}
    jsoncon['role'] = role
    jsoncon['content'] = content
    text.append(jsoncon)
    return text

def get_spark_response(data):
    # 配置Spark API
    appid = "your appid"
    api_key = "your api_key"
    api_secret = "your api_secret"
    Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat"

    # 构造请求函数
    if data.if_list:
        # 多轮对话
        prompt = data.prompt
    else:
        prompt = getText("user", data.prompt)

    # 调用SparkApiSelf.py中的main函数
    response = SparkApiSelf.main(appid, api_key, api_secret, Spark_url, prompt, data.max_tokens, data.temperature)
    return response

@app.post("/spark/")
# @app.post("/spark/")的作用是将POST请求映射到/spark/路径上
async def spark(data: Data):
# async def spark(data: Data)的作用是定义一个异步函数，接收Data类型的数据
    response = get_spark_response(data)
    print(response) # 打印返回的结果
    return response