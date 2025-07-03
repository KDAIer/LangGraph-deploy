import os
from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from agent.llm_client import LLMClient

# 在已有段落的指定位置插入文献/网页引用链接（utils.py 中的函数）
from agent.utils import ( 
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)

# LLMClient 是一个封装了 LLM 调用的客户端，支持多种模型和配置
llm = LLMClient() 

# 生成初始查询：根据输入，它使用 deepseek-v3 模型生成一组初始搜索查询
# 自动编写若干检索关键词，供后续做真正或模拟搜索使用
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph 节点根据用户的问题生成搜索查询

    Args:
        state: 当前图状态，包含用户消息和初始搜索查询计数
        config: 可运行配置，包含查询生成模型和初始查询数量
    Returns:
        包含状态更新的字典，其中包含已生成查询的 search_query 键
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # Format the prompt
    formatted_prompt = query_writer_instructions.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )

    # 调用 LLMClient 的 query 方法，传入格式化的提示和模型
    raw = llm.query(
        formatted_prompt,
        model_name="deepseek-v3",
        temperature=1.0,
        deployment="ali",
        system_message="你是一个结构化响应助手，只返回纯 JSON，不要任何 code fence、注释或多余文字。"
    )
    print("generate_query raw repr:", repr(raw))

    from agent.utils import strip_code_fence
    import json

    raw_clean = strip_code_fence(raw)
    print("generate_query cleaned JSON:", raw_clean)
    if not raw_clean:
        # 如果 LLM 真没回，至少用用户输入当做单条 query
        fallback = get_research_topic(state["messages"])
        print(f"LLM 无返回，fallback to: {fallback}")
        return {"search_query": [fallback]}

    try:
        parsed = json.loads(raw_clean)
    except json.JSONDecodeError as e:
        # JSON 解析失败时，也 fallback
        print(f"JSONDecodeError: {e}; raw_clean repr: {repr(raw_clean)}")
        return {"search_query": [raw_clean]}

    # —— 兼容多种格式 —— 
    if isinstance(parsed, list):
        queries = parsed
    elif "queries" in parsed:
        queries = parsed["queries"]
    elif "query" in parsed:
        queries = [parsed["query"]]
    else:
        queries = [line.strip() for line in raw_clean.splitlines() if line.strip()]

    return {"search_query": queries}

# 把上一步生成的多个 search_query 拆成 N 条并行任务，分别发给 web_research 节点
def continue_to_web_research(state: QueryGenerationState):
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


# 网络研究：对于每个查询，它使用 deepseek-v3 模型和网页搜索工具（//mysql数据库）来查找相关的网页
def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    # Configure
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    # Uses the google genai client as the langchain client doesn't return grounding metadata
    # response = genai_client.models.generate_content(
    #     model=configurable.query_generator_model,
    #     contents=formatted_prompt,
    #     config={
    #         "tools": [{"google_search": {}}],
    #         "temperature": 0,
    #     },
    # )
    # # resolve the urls to short urls for saving tokens and time
    # resolved_urls = resolve_urls(
    #     response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
    # )
    # # Gets the citations and adds them to the generated text
    # citations = get_citations(response, resolved_urls)
    # modified_text = insert_citation_markers(response.text, citations)
    # sources_gathered = [item for citation in citations for item in citation["segments"]]
    # 使用 LLMClient 获取“假想”的搜索摘要
    raw = llm.query(
        formatted_prompt,
        model_name="deepseek-v3",
        temperature=0,
        deployment="ali",
    )
    # 如果需要引用标记，可按原逻辑解析 raw 中的 citation 字段；此处简化为直接返回文本
    modified_text = raw
    # 这里我们不再维护 grounding 元数据
    sources_gathered = []
 
    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }

# 反思与知识差距分析：代理会分析搜索结果，以确定信息是否充足或是否存在知识差距
# 使用 deepseek-v3 模型来判断是否需要进一步研究
def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    configurable = Configuration.from_runnable_config(config)
    # 统计轮数
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    formatted_prompt = reflection_instructions.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )

    # 调用 LLMClient
    raw = llm.query(
        formatted_prompt,
        model_name="deepseek-v3",
        temperature=1.0,
        deployment="ali",
        system_message="请严格以 JSON 格式返回 is_sufficient, knowledge_gap 和 follow_up_queries 字段"
    )

    from agent.utils import strip_code_fence
    cleaned = strip_code_fence(raw)

    # 打印清洗后的内容，便于调试
    print("reflection cleaned JSON:", cleaned)

    # 尝试解析 JSON，失败时安全降级
    import json
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("JSONDecodeError:", e, "原始清洗后文本：", cleaned)
        parsed = {}

    is_sufficient     = parsed.get("is_sufficient", False)
    knowledge_gap     = parsed.get("knowledge_gap", "")
    follow_up_queries = parsed.get("follow_up_queries", [])
    return {
        "is_sufficient": is_sufficient, # 判断目前收集到的摘要是否已足够回答用户
        "knowledge_gap": knowledge_gap, # 当前知识是否存在差距
        "follow_up_queries": follow_up_queries, # 后续查询列表
        "research_loop_count": state["research_loop_count"], # 当前研究轮数
        "number_of_ran_queries": len(state["search_query"]), # 已执行查询数量
    }

# 迭代细化：如果发现差距或信息不足，它会生成后续查询并重复网络研究和反思步骤（最多配置的最大循环次数）
def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:

    configurable = Configuration.from_runnable_config(config)

    # 达到最大研究轮数或已满足条件时，直接返回 finalize_answer
    max_research_loops = ( 
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]

# Finalize the answer: 该节点将收集到的研究结果和引用整合成最终的研究报告，并添加适当的引用标记
def finalize_answer(state: OverallState, config: RunnableConfig):

    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    raw = llm.query(
        formatted_prompt,
        model_name="deepseek-v3",
        temperature=0,
        deployment="ali",
    )

    # 这里模拟替换短链为原始 URL
    class Dummy: pass
    result = Dummy()
    result.content = raw

    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
