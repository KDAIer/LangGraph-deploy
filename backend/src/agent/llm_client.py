from typing import Optional, Dict, Any, List, Tuple, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from functools import lru_cache
import logging
import openai
import time
import asyncio
import os
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from functools import lru_cache
import logging
import openai
import time
import asyncio
from langchain.callbacks.tracers import LangChainTracer
from langchain_core.callbacks import CallbackManager


class LLMClient:
    """高性能大模型请求客户端"""

    def __init__(self):
        # 配置日志
        self.logger = logging.getLogger("LLMClient")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        # 缓存模型实例
        self._model_cache = {}
        # 缓存embedding模型实例
        self._embedding_cache = {}
        # 追踪性能数据
        self.performance_stats = {
            "total_requests": 0,
            "success_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "avg_response_time": 0,
            "embedding_requests": 0,
            "embedding_success": 0,
            "embedding_failed": 0,
        }
        self.max_retries = 3  # 最大重试次数
        self.retry_delay = 1  # 重试延迟(秒)
        self.retryable_errors = (
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError
        )
        self.tracer = LangChainTracer()
        self.callback_manager = CallbackManager([self.tracer])

    @lru_cache(maxsize=10)
    def _get_model(self, model_name: str, api_key: str, api_base: str, temperature: float) -> ChatOpenAI:
        """获取模型实例，使用lru缓存避免重复创建相同参数的模型"""
        return ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base=api_base,
            model_name=model_name,
            temperature=temperature,
            timeout=60,  # 设置超时时间
            max_retries=2,  # 设置重试次数
            callback_manager=self.callback_manager,
            verbose=True,  
        )

    @lru_cache(maxsize=5)
    def _get_embedding_model(self, model_name: str, api_key: str, api_base: str) -> OpenAIEmbeddings:
        """获取embedding模型实例，使用lru缓存避免重复创建相同参数的模型"""
        return OpenAIEmbeddings(
            openai_api_key=api_key,
            openai_api_base=api_base,
            model=model_name,
            timeout=30,
            max_retries=2,
        )

    def get_embeddings(self,
                       texts: Union[str, List[str]],
                       model_name: str = "qwen-1.5b",
                       deployment: str = "embedding") -> List[List[float]]:
        """
        获取文本的嵌入向量

        Args:
            texts: 单个文本字符串或文本列表
            model_name: 嵌入模型名称
            deployment: 部署类型

        Returns:
            嵌入向量列表
        """
        self.performance_stats["embedding_requests"] += 1
        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # 获取API配置
                api_config = self._get_api_config(deployment)
                model_name = api_config.get("default_model", model_name) if model_name is None else model_name

                # 获取embedding模型实例
                embedding_model = self._get_embedding_model(
                    model_name,
                    api_config["api_key"],
                    api_config["api_base"]
                )

                # 处理单个文本或文本列表
                if isinstance(texts, str):
                    embeddings = embedding_model.embed_query(texts)
                    embeddings = [embeddings]  # 转为列表的列表格式保持一致性
                else:
                    embeddings = embedding_model.embed_documents(texts)

                # 更新性能统计
                self.performance_stats["embedding_success"] += 1
                elapsed = time.time() - start_time
                self.logger.debug(
                    f"Embedding successful: {elapsed:.2f}s, model: {model_name}, deployment: {deployment}")

                return embeddings

            except self.retryable_errors as e:
                last_error = e
                self.performance_stats["embedding_failed"] += 1
                wait_time = self.retry_delay * (attempt + 1)
                self.logger.warning(
                    f"Embedding attempt {attempt + 1}/{self.max_retries} failed. "
                    f"Retrying in {wait_time}s. Error: {str(e)}"
                )
                time.sleep(wait_time)

            except Exception as e:
                self.performance_stats["embedding_failed"] += 1
                self.logger.error(f"Non-retryable embedding error: {str(e)}")
                raise

        self.logger.error(f"Max retries ({self.max_retries}) exceeded for embedding. Last error: {str(last_error)}")
        raise last_error if last_error else Exception("Unknown error occurred during embedding")

    async def aget_embeddings(self,
                              texts: Union[str, List[str]],
                              model_name: str = "qwen-1.5b",
                              deployment: str = "embedding") -> List[List[float]]:
        """
        异步获取文本的嵌入向量

        Args:
            texts: 单个文本字符串或文本列表
            model_name: 嵌入模型名称
            deployment: 部署类型

        Returns:
            嵌入向量列表
        """
        self.performance_stats["embedding_requests"] += 1
        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # 获取API配置
                api_config = self._get_api_config(deployment)
                model_name = api_config.get("default_model", model_name) if model_name is None else model_name

                # 获取embedding模型实例
                embedding_model = self._get_embedding_model(
                    model_name,
                    api_config["api_key"],
                    api_config["api_base"]
                )

                # 处理单个文本或文本列表
                if isinstance(texts, str):
                    embeddings = await embedding_model.aembed_query(texts)
                    embeddings = [embeddings]  # 转为列表的列表格式保持一致性
                else:
                    embeddings = await embedding_model.aembed_documents(texts)

                # 更新性能统计
                self.performance_stats["embedding_success"] += 1
                elapsed = time.time() - start_time
                self.logger.debug(
                    f"Async embedding successful: {elapsed:.2f}s, model: {model_name}, deployment: {deployment}")

                return embeddings

            except self.retryable_errors as e:
                last_error = e
                self.performance_stats["embedding_failed"] += 1
                wait_time = self.retry_delay * (attempt + 1)
                self.logger.warning(
                    f"Async embedding attempt {attempt + 1}/{self.max_retries} failed. "
                    f"Retrying in {wait_time}s. Error: {str(e)}"
                )
                await asyncio.sleep(wait_time)

            except Exception as e:
                self.performance_stats["embedding_failed"] += 1
                self.logger.error(f"Non-retryable async embedding error: {str(e)}")
                raise

        self.logger.error(
            f"Max retries ({self.max_retries}) exceeded for async embedding. Last error: {str(last_error)}")
        raise last_error if last_error else Exception("Unknown error occurred during async embedding")

    async def abatch_get_embeddings(self,
                                    text_batches: List[List[str]],
                                    model_name: str = "qwen-1.5b",
                                    deployment: str = "embedding",
                                    concurrency: int = 3) -> List[List[List[float]]]:
        """
        批量异步获取嵌入向量，每个批次单独处理

        Args:
            text_batches: 批次文本列表的列表，每个批次会单独发送请求
            model_name: 嵌入模型名称
            deployment: 部署类型
            concurrency: 最大并发数

        Returns:
            每个批次的嵌入向量列表的列表
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded_embedding(texts):
            async with semaphore:
                return await self.aget_embeddings(texts, model_name, deployment)

        tasks = [_bounded_embedding(batch) for batch in text_batches]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        计算两个嵌入向量之间的余弦相似度

        Args:
            embedding1: 第一个嵌入向量
            embedding2: 第二个嵌入向量

        Returns:
            余弦相似度 (-1 到 1 之间的值，1 表示完全相同)
        """
        # 将列表转换为numpy数组
        v1 = np.array(embedding1)
        v2 = np.array(embedding2)

        # 计算余弦相似度
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        # 避免除零错误
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        similarity = dot_product / (norm_v1 * norm_v2)
        return float(similarity)

    def query(self,
              query: Union[str, List[Dict[str, str]]],
              model_name: str = "deepseek-v3",
              temperature: float = 1.,
              deployment: str = "ali",
              system_message: str = None) -> str:
        """
        同步查询大模型

        Args:
            query: 字符串格式的查询或消息列表
            model_name: 模型名称
            temperature: 温度参数
            deployment: 部署类型 (dashscope, local, qwen)
            system_message: 可选的系统消息

        Returns:
            模型回复内容
        """
        self.performance_stats["total_requests"] += 1
        start_time = time.time()
        last_error = None
        for attempt in range(self.max_retries):
            self.performance_stats["total_requests"] += 1
            start_time = time.time()

            try:
                # 根据部署类型选择API配置
                api_config = self._get_api_config(deployment)
                model_name = api_config.get("default_model", model_name) if model_name is None else model_name

                # 获取模型实例
                model = self._get_model(
                    model_name,
                    api_config["api_key"],
                    api_config["api_base"],
                    temperature
                )

                # 处理请求格式
                if isinstance(query, str):
                    if system_message:
                        messages = [
                            SystemMessage(content=system_message),
                            HumanMessage(content=query)
                        ]
                        response = model.invoke(messages)
                    else:
                        response = model.invoke(query)
                else:
                    # 假设query是消息列表格式
                    response = model.invoke(query)

                self.performance_stats["success_requests"] += 1

                # 记录性能数据
                elapsed = time.time() - start_time
                self._update_performance_metrics(elapsed)

                self.logger.debug(f"Query successful: {elapsed:.2f}s, model: {model_name}, deployment: {deployment}")
                return response.content
            except self.retryable_errors as e:
                last_error = e
                self.performance_stats["failed_requests"] += 1
                wait_time = self.retry_delay * (attempt + 1)
                self.logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed. "
                    f"Retrying in {wait_time}s. Error: {str(e)}"
                )
                time.sleep(wait_time)


            except Exception as e:
                self.performance_stats["failed_requests"] += 1
                self.logger.error(f"Non-retryable error: {str(e)}")
                raise

        self.logger.error(f"Max retries ({self.max_retries}) exceeded. Last error: {str(last_error)}")
        raise last_error if last_error else Exception("Unknown error occurred")

    async def aquery(self,
                     query: Union[str, List[Dict[str, str]]],
                     model_name: str = "deepseek-v3",
                     temperature: float = 1.,
                     deployment: str = "ali",
                     system_message: str = None) -> str:
        """
        异步查询大模型

        Args:
            query: 字符串格式的查询或消息列表
            model_name: 模型名称
            temperature: 温度参数
            deployment: 部署类型 (dashscope, local, qwen)
            system_message: 可选的系统消息

        Returns:
            模型回复内容
        """
        self.performance_stats["total_requests"] += 1
        last_error = None
        for attempt in range(self.max_retries):
            self.performance_stats["total_requests"] += 1
            start_time = time.time()

            try:
                # 根据部署类型选择API配置
                api_config = self._get_api_config(deployment)
                model_name = api_config.get("default_model", model_name) if model_name is None else model_name

                # 获取模型实例
                model = self._get_model(
                    model_name,
                    api_config["api_key"],
                    api_config["api_base"],
                    temperature
                )

                # 处理请求格式
                if isinstance(query, str):
                    if system_message:
                        messages = [
                            SystemMessage(content=system_message),
                            HumanMessage(content=query)
                        ]
                        response = await model.ainvoke(messages)
                    else:
                        response = await model.ainvoke(query)
                else:
                    # 假设query是消息列表格式
                    response = await model.ainvoke(query)

                self.performance_stats["success_requests"] += 1

                # 记录性能数据
                elapsed = time.time() - start_time
                self._update_performance_metrics(elapsed)

                self.logger.debug(
                    f"Async query successful: {elapsed:.2f}s, model: {model_name}, deployment: {deployment}")
                return response.content
            except self.retryable_errors as e:
                last_error = e
                self.performance_stats["failed_requests"] += 1
                wait_time = self.retry_delay * (attempt + 1)
                self.logger.warning(
                    f"Async attempt {attempt + 1}/{self.max_retries} failed. "
                    f"Retrying in {wait_time}s. Error: {str(e)}"
                )
                await asyncio.sleep(wait_time)
            except Exception as e:
                self.performance_stats["failed_requests"] += 1
                self.logger.error(f"Non-retryable async error: {str(e)}")

                raise

    def batch_query(self,
                    queries: List[str],
                    model_name: str = "deepseek-v3",
                    temperature: float = 1.,
                    deployment: str = "ali") -> List[str]:
        """
        批量同步查询

        Args:
            queries: 查询字符串列表
            model_name: 模型名称
            temperature: 温度参数
            deployment: 部署类型

        Returns:
            回复内容列表
        """
        results = []
        for query in queries:
            results.append(self.query(query, model_name, temperature, deployment))
        return results

    async def abatch_query(self,
                           queries: List[str],
                           model_name: str = "deepseek-v3",
                           temperature: float = 1.,
                           deployment: str = "ali",
                           concurrency: int = 5) -> List[str]:
        """
        批量异步查询，支持控制并发数量

        Args:
            queries: 查询字符串列表
            model_name: 模型名称
            temperature: 温度参数
            deployment: 部署类型
            concurrency: 最大并发数量

        Returns:
            回复内容列表
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded_query(query):
            async with semaphore:
                return await self.aquery(query, model_name, temperature, deployment)

        tasks = [_bounded_query(query) for query in queries]
        return await asyncio.gather(*tasks, return_exceptions=True)


    def _get_api_config(self, deployment: str) -> Dict[str, str]:
        configs = {
            "ali": {
                # 从环境变量读取 DashScope Key
                "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
                "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "default_model": "deepseek-v3",
            },
            "local": {
                "api_key": os.getenv("LOCAL_API_KEY", ""),
                "api_base": os.getenv("LOCAL_API_BASE", "http://10.13.10.252:3001/v1"),
                "default_model": "cogito-qwen",
            },
            "embedding": {
                "api_key": os.getenv("EMBED_API_KEY", ""),
                "api_base": os.getenv("EMBED_API_BASE", "http://10.120.17.167:7900/v1"),
                "default_model": "qwen-1.5b",
            },
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "api_base": "https://api.openai.com/v1",
                "default_model": "gpt-3.5-turbo",
            },
        }
        if deployment not in configs:
            raise ValueError(f"Unknown deployment type: {deployment}")
        # 简易验证
        key = configs[deployment]["api_key"]
        if not key:
            raise ValueError(f"Missing API key for deployment '{deployment}'. "
                             f"Please set the corresponding env var.")
        return configs[deployment]


    def _update_performance_metrics(self, elapsed_time: float) -> None:
        """更新性能指标"""
        total_success = self.performance_stats["success_requests"]
        current_avg = self.performance_stats["avg_response_time"]

        # 更新平均响应时间
        if total_success > 1:
            self.performance_stats["avg_response_time"] = (current_avg * (
                    total_success - 1) + elapsed_time) / total_success
        else:
            self.performance_stats["avg_response_time"] = elapsed_time

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计数据"""
        return self.performance_stats


if __name__ == '__main__':
    # 测试embedding 模型

    llm_tool = LLMClient()
    start_time = time.time()
    res = llm_tool.query('你是谁', model_name='qwen-plus', deployment='ali')
    print(res)