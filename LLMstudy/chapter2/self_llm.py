"""
    在LangChain的基础上封装的项目类
"""
from langchain.llms.base import LLM
from typing import Dict, Any, Mapping
# Dict: 字典类型
# Any: 任意类型
# Mapping: 映射类型
from pydantic import Field
# Field: 字段

# 项目类
class Self_LLM(LLM):
    """
        项目类
        继承自LLM
    """
    url : str = None
    # 默认选用zhipu模型
    model_name : str = "zhipu"
    # 访问的时延上限
    request_timeout : float = 10.0
    # API key
    api_key : str = None 

    # 必备的可选参数
    model_kwargs : Dict[str, Any] = Field(default_factory=dict)

    # 定义一个返回默认参数的方法
    @property
    # property的作用是将一个方法变成一个属性
    def _default_params(self) -> Dict[str, Any]:
        """
            返回默认参数
        """
        normal_params = {
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
            
        }
        return {**normal_params, **self.model_kwargs}
        # **normal_params: 将normal_params中的所有键值对解包到字典中
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
            get the identifying params
        """
        return {**{"model_name": self.model_name}, **self.model_kwargs}