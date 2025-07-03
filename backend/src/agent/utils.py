from typing import Any, Dict, List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
import re

def strip_code_fence(text: str) -> str:
    """
    去掉多余的 ``` 或 ```json fences，只留下内部内容。
    如果没 fence，则返回原字符串 strip() 后的结果。
    """
    text = text.strip()
    pattern = r"^```(?:json)?\s*([\s\S]+?)\s*```$"
    m = re.match(pattern, text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text

# 从 LangGraph 的消息流中提炼“研究主题”文本
def get_research_topic(messages: List[AnyMessage]) -> str:
    if len(messages) == 1: # 如果只有一条消息，直接取它的 content 作为主题
        research_topic = messages[-1].content
    else:
        research_topic = ""
        # 遍历所有消息，提取人类和 AI 的内容
        # 人类消息前缀为 "User: "，AI 消息前缀为 "Assistant: "
        # 这样可以确保研究主题包含所有相关对话内容
        for message in messages: 
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


# 把原本很长的网页或文档 URI 映射成统一短链，方便在报告中引用
def resolve_urls(urls_to_resolve: List[Any], id: int) -> Dict[str, str]:
    prefix = f"https://extractresearch.frame.com/id/"
    urls = [site.web.uri for site in urls_to_resolve]

    resolved_map = {}
    # 按出现顺序给每个唯一 URL 分配短链
    for idx, url in enumerate(urls):
        if url not in resolved_map:
            resolved_map[url] = f"{prefix}{id}-{idx}"

    return resolved_map

# 在已有段落的指定位置插入文献/网页引用链接
def insert_citation_markers(text, citations_list):
    """

    Args:
        text: 原文字符串。
        citations_list: 每条引用信息包含: 
            start_index、end_index: 标记在原文中的字符区间。
            segments: 列表，每个元素有 label(链接文本)、short_url(跳转短链)。

    Returns:
        str: The text with citation markers inserted.
    """

    # 按 end_index 从大到小排序，避免插入后影响前面索引
    sorted_citations = sorted(
        citations_list, key=lambda c: (c["end_index"], c["start_index"]), reverse=True
    )

    # 对每个引用，生成诸如 [来源标签](短链) 的 Markdown 链接串，插入到 end_index 位置
    modified_text = text
    for citation_info in sorted_citations:
        end_idx = citation_info["end_index"]
        marker_to_insert = ""
        for segment in citation_info["segments"]:
            marker_to_insert += f" [{segment['label']}]({segment['short_url']})"
        # Insert the citation marker at the original end_idx position
        modified_text = (
            modified_text[:end_idx] + marker_to_insert + modified_text[end_idx:]
        )

    return modified_text


# 从 LLM 响应的 grounding_metadata 中提取引用所需信息
# grounding_metadata 是模型提供的支持性证据，包含原文片段和引用的网页信息
def get_citations(response, resolved_urls_map):

    citations = []

    # 没有引用
    if not response or not response.candidates:
        return citations

    # 只处理第一个候选项
    candidate = response.candidates[0]
    if (
        not hasattr(candidate, "grounding_metadata")
        or not candidate.grounding_metadata
        or not hasattr(candidate.grounding_metadata, "grounding_supports")
    ):
        return citations

    # 这是模型给出的“支持性证据”列表。每个 support 包含两个关键信息块：
    # 1. segment: 该支持信息在原文中的起止位置
    # 2. grounding_chunk_indices: 引用的具体段落或网页片段
    for support in candidate.grounding_metadata.grounding_supports:
        citation = {}

        if not hasattr(support, "segment") or support.segment is None:
            continue  

        start_index = (
            support.segment.start_index
            if support.segment.start_index is not None
            else 0
        )

        # 如果没有 end_index，跳过这个支持项
        # end_index 是必须的，因为它定义了引用的结束位置
        if support.segment.end_index is None:
            continue 

        # 构建 citation 对象
        citation["start_index"] = start_index
        citation["end_index"] = support.segment.end_index

        citation["segments"] = []
        if (
            hasattr(support, "grounding_chunk_indices")
            and support.grounding_chunk_indices
        ):
            for ind in support.grounding_chunk_indices:
                try:
                    chunk = candidate.grounding_metadata.grounding_chunks[ind]
                    resolved_url = resolved_urls_map.get(chunk.web.uri, None)
                    citation["segments"].append(
                        {
                            "label": chunk.web.title.split(".")[:-1][0],
                            "short_url": resolved_url,
                            "value": chunk.web.uri,
                        }
                    )
                except (IndexError, AttributeError, NameError):
                    pass
        citations.append(citation)
    return citations
