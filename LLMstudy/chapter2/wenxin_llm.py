"""
    自定义的wenxin_llm模块
"""
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional, Dict, Union, Tuple
from pydantic import Field
from LLMstudy.chapter2.self_llm import Self_LLM
import json
import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun

# 获取access_token
def get_access_token(api_key:str, secret_key:str):
    """
        使用api_key和secret_key获取access_token
    """
    # 获取access_token的url
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
    # 设置POST访问
    pyload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json'
        }
    # 通过post请求获取access_token
    response = requests.request("POST", url, headers=headers, data=pyload)
    return response.json().get("access_token")

# 文心大模型类
class Wenxin_LLM(Self_LLM):
    """
        文心大模型类
    """
    # url
    url : str = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token={}"

    # secret_key
    secret_key : str = None
    # access_token
    access_token : str = None

    # 初始化access_token
    def init_access_token(self):
        """
            初始化access_token
        """
        if self.api_key != None and self.secret_key != None:
            try:
                self.access_token = get_access_token(self.api_key, self.secret_key)
            except Exception as e:
                print(e)
                print("获取access_token失败")
        else:
            print("api_key和secret_key不能为空")

    def _call(self, prompt : str, stop : Optional[List[str]] = None,
              run_manager : Optional[CallbackManagerForLLMRun] = None,
              **kwargs : Any):
        # 如果access_token为空，初始化access_token
        if self.access_token == None:
            self.init_access_token()
        # API 调用 url
        url = self.url.format(self.access_token)
        # 配置post参数
        payload = json.dumps({
            "message":[
                {
                    "role":"user",
                    "content": "{}".format(prompt)
                }
            ],
            'temperature': self.temperature
        })

        headers = {
            'Content-Type': 'application/json',
        }

        # 发起post请求
        response = requests.request("POST", url, headers=headers, data=payload, timeout=self.request_timeout)

        if response.status_code == 200:
            # 返回一个json字符串
            js = json.loads(response.text)
            # print(js)
            return js['result']
        else:
            print("请求失败")
            return None

        @property
        def _llm_type(self) -> str:
            return "wenxin_llm"