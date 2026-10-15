---
title: 附录E 从零开始构建智能体
tags:
  - Agent
  - Tool Use
  - ReAct
  - Memory
  - Planning
  - hello-agents
description: 从零开始构建智能体. 整合 datawhalechina/hello-agents 核心内容. 涵盖智能体原理、工具调用、记忆、规划、多智能体协作
last_updated: 2026-09-11
status: ✅ v1.0 - 整合 hello-agents
chapter_id: appendix-e
---

# 附录E 从零开始构建智能体

> 本附录整合自 [datawhalechina/hello-agents](https://github.com/datawhalechina/hello-agents) (从零开始构建智能体). 涵盖智能体的核心原理与实践.

---

## 1. 什么是智能体 (Agent)

**智能体** = 能感知环境、做出决策、执行动作的 AI 系统. 区别于普通 LLM:

| 维度 | LLM | Agent |
| :--- | :--- | :--- |
| **输入** | 单次 prompt | 多轮 + 工具调用 |
| **输出** | 文本响应 | 文本 + 动作 |
| **目标** | 回答问题 | 完成多步任务 |
| **能力** | 纯生成 | 推理 + 工具 + 记忆 |
| **示例** | "什么是 RAG?" | "研究 XXX 主题, 写报告" |

---

## 2. Agent 核心架构

```text
用户请求
   ↓
┌─────────────────────────────────┐
│  Agent Loop                      │
│  ┌──────────┐                   │
│  │ Perceive │ ←── 环境/工具反馈  │
│  └─────┬────┘                   │
│        ↓                        │
│  ┌──────────┐                   │
│  │   Think  │ ←── LLM (推理)    │
│  └─────┬────┘                   │
│        ↓                        │
│  ┌──────────┐                   │
│  │   Act    │ → 工具调用/响应   │
│  └─────┬────┘                   │
│        ↓                        │
│     Loop until done             │
└─────────────────────────────────┘
   ↓
最终响应
```

---

## 3. Agent 核心循环 (ReAct)

### 3.1 ReAct 原理

**ReAct = Reasoning + Acting** (Yao et al., 2022)

```python
from openai import OpenAI

class ReActAgent:
    def __init__(self, tools, llm_client, max_steps=10):
        self.tools = tools  # 工具列表 (字典: name -> func)
        self.llm = llm_client
        self.max_steps = max_steps
    
    def run(self, query):
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": query}
        ]
        
        for step in range(self.max_steps):
            # 1. LLM 推理 + 决定动作
            response = self.llm.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=self._format_tools()
            )
            
            msg = response.choices[0].message
            
            # 2. 如果 LLM 调用工具
            if msg.tool_calls:
                messages.append(msg)
                
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    # 3. 执行工具
                    result = self.tools[func_name](**func_args)
                    
                    # 4. 工具结果回填
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })
            else:
                # 5. LLM 直接回答, 结束
                messages.append(msg)
                return msg.content
        
        return "达到最大步数限制"
    
    def _system_prompt(self):
        return f"""你是一个 ReAct Agent. 按以下格式思考:

Thought: [你的思考]
Action: 工具名(参数)
Observation: [工具结果]
... (循环)
Thought: 我现在可以回答了
Final Answer: [最终答案]

可用工具:
{self._format_tools_descriptions()}
"""
```

### 3.2 完整可运行示例

```python
import json
from openai import OpenAI

# 1. 定义工具
def get_weather(city: str) -> str:
    """查询天气"""
    weather_data = {
        "北京": "晴, 25°C, 风力 2 级",
        "上海": "多云, 28°C, 风力 3 级",
        "广州": "雨, 30°C, 风力 4 级",
    }
    return weather_data.get(city, "未知城市")

def calculator(expr: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expr))
    except:
        return "计算错误"

# 2. 工具描述 (OpenAI Function Calling 格式)
tools_spec = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expr": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expr"]
            }
        }
    }
]

tools_map = {
    "get_weather": get_weather,
    "calculator": calculator
}

# 3. Agent Loop
client = OpenAI()

def run_agent(query):
    messages = [
        {"role": "system", "content": "你是一个能调用工具的助手。"},
        {"role": "user", "content": query}
    ]
    
    for step in range(8):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools_spec,
            tool_choice="auto"
        )
        
        msg = response.choices[0].message
        messages.append(msg)
        
        if not msg.tool_calls:
            return msg.content
        
        # 执行所有工具调用
        for tool_call in msg.tool_calls:
            func = tools_map[tool_call.function.name]
            args = json.loads(tool_call.function.arguments)
            result = func(**args)
            print(f"  → {tool_call.function.name}({args}) = {result}")
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
    
    return "达到最大步数"

# 4. 测试
query = "北京今天天气如何? 如果 25 度, 那相当于多少华氏度? (华氏度 = 摄氏度 × 9/5 + 32)"
answer = run_agent(query)
print(f"\n最终回答: {answer}")
# 输出:
#   → get_weather({'city': '北京'}) = 晴, 25°C, 风力 2 级
#   → calculator({'expr': '25 * 9/5 + 32'}) = 77.0
# 最终回答: 北京今天是晴, 25°C, 风力 2 级, 相当于 77°F.
```

---

## 4. Agentic 设计模式

### 4.1 Reflection (自我反思)

Agent 审视自己的输出, 改进。

```python
def reflective_agent(task):
    # 1. 初版生成
    draft = llm.generate(task)
    
    # 2. 自我批评
    critique = llm.generate(f"""
请批评以下输出, 指出问题:
{draft}
""")
    
    # 3. 改进版
    improved = llm.generate(f"""
基于批评改进:
原版: {draft}
批评: {critique}

改进版:
""")
    
    return improved
```

### 4.2 Planning (规划)

将复杂任务分解为子任务。

```python
def plan_and_execute(goal):
    # 1. 生成计划
    plan = llm.generate(f"""
为以下目标生成执行计划:
目标: {goal}

输出 JSON 列表:
[
    {{"step": 1, "task": "...", "tool": "..."}},
    ...
]
""")
    plan = json.loads(plan)
    
    # 2. 执行每步
    results = []
    context = ""
    for step in plan:
        result = execute_step(step, context)
        context += f"\n步骤 {step['step']} 结果: {result}"
        results.append(result)
    
    # 3. 汇总
    final = llm.generate(f"""
基于以下步骤结果, 总结最终答案:
{context}
""")
    return final
```

### 4.3 Tool Use (工具使用)

```python
# LangChain 工具定义
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI

@tool
def search_wikipedia(query: str) -> str:
    """搜索维基百科"""
    # 实际调用 Wikipedia API
    return f"Wikipedia 搜索结果: {query}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件"""
    return f"邮件已发送到 {to}"

# 创建 Agent
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools=[search_wikipedia, send_email],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 运行
agent.run("搜索'人工智能'并把摘要发到 user@example.com")
```

### 4.4 Multi-Agent 协作

详见 [算法与AI/Agent技术/第06章-多智能体系统](第06章-多智能体系统.md)。

```python
# CrewAI 示例
from crewai import Agent, Task, Crew

# 定义 Agent
researcher = Agent(
    role='研究员',
    goal='深入研究指定主题',
    backstory='你是 AI 研究专家'
)

writer = Agent(
    role='作家',
    goal='把研究内容写成清晰文章',
    backstory='你是技术作家'
)

# 定义任务
research_task = Task(
    description='研究 LLM 最新进展',
    agent=researcher,
    expected_output='2000 字研究报告'
)

write_task = Task(
    description='基于研究写一篇博客',
    agent=writer,
    expected_output='1500 字博客'
)

# 创建 Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    verbose=True
)

result = crew.kickoff()
```

---

## 5. Agent 评估

### 5.1 评估指标

| 指标 | 含义 | 计算 |
| :--- | :--- | :--- |
| **任务完成率** | 完成的任务比例 | 成功任务数 / 总任务数 |
| **工具调用准确率** | 工具调用正确率 | 正确调用 / 总调用 |
| **步数效率** | 完成任务的步数 | 与最小步数比 |
| **成本** | 完成任务的总成本 | token × 价格 |
| **鲁棒性** | 错误情况下恢复 | 失败重试成功率 |

### 5.2 AgentBench

```python
# 使用 AgentBench 评估
from agentbench import AgentBench

bench = AgentBench()
results = bench.evaluate(
    agent=my_agent,
    tasks=bench.load_tasks("gaia"),
    n_trials=3
)
print(f"任务完成率: {results.success_rate:.2%}")
```

---

## 6. 实战: 构建一个文件管理 Agent

```python
import os
import shutil
from pathlib import Path

class FileManagerAgent:
    """可以管理文件的 Agent"""
    
    TOOLS = {
        "list_files": {
            "description": "列出目录文件",
            "parameters": {"path": "string"}
        },
        "read_file": {
            "description": "读取文件内容",
            "parameters": {"path": "string"}
        },
        "write_file": {
            "description": "写入文件",
            "parameters": {"path": "string", "content": "string"}
        },
        "move_file": {
            "description": "移动文件",
            "parameters": {"src": "string", "dst": "string"}
        },
        "delete_file": {
            "description": "删除文件",
            "parameters": {"path": "string"}
        }
    }
    
    def __init__(self, workspace):
        self.workspace = Path(workspace)
    
    def execute_tool(self, name, args):
        if name == "list_files":
            path = self.workspace / args['path']
            return str(list(path.iterdir()) if path.exists() else "路径不存在")
        elif name == "read_file":
            path = self.workspace / args['path']
            return path.read_text() if path.exists() else "文件不存在"
        elif name == "write_file":
            path = self.workspace / args['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args['content'])
            return f"已写入 {path}"
        elif name == "move_file":
            src = self.workspace / args['src']
            dst = self.workspace / args['dst']
            shutil.move(str(src), str(dst))
            return f"已移动 {src} → {dst}"
        elif name == "delete_file":
            path = self.workspace / args['path']
            if path.exists():
                path.unlink()
                return f"已删除 {path}"
            return "文件不存在"
        return "未知工具"
    
    def run(self, instruction):
        messages = [
            {"role": "system", "content": f"你是文件管理 Agent, 工作目录: {self.workspace}\n可用工具: {self.TOOLS}"},
            {"role": "user", "content": instruction}
        ]
        
        for step in range(15):
            response = self.llm.chat(messages, tools=list(self.TOOLS.keys()))
            
            if not response.tool_calls:
                return response.content
            
            for tc in response.tool_calls:
                result = self.execute_tool(tc.name, tc.args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        
        return "达到最大步数"

# 使用
agent = FileManagerAgent("./workspace")
result = agent.run("在 workspace 下创建 notes 文件夹, 里面写一个 hello.md, 内容是 'Hello, Agent!'")
print(result)
```

---

## 7. 智能体记忆系统

详见 [算法与AI/Agent技术/第07章-记忆与上下文工程](第07章-记忆与上下文工程.md).

```python
class AgentMemory:
    """Agent 的三层记忆"""
    def __init__(self):
        # 短期记忆 (当前对话)
        self.short_term = []
        # 长期记忆 (向量化)
        self.long_term = VectorStore()
        # 情节记忆 (关键事件)
        self.episodic = []
    
    def remember(self, info, importance=0.5):
        if importance > 0.8:
            # 重要 → 长期记忆
            self.long_term.add(info)
        else:
            self.short_term.append(info)
    
    def recall(self, query, top_k=3):
        # 从长期记忆检索
        return self.long_term.search(query, k=top_k)
    
    def forget_short_term(self):
        # 清理过期的短期记忆
        if len(self.short_term) > 20:
            self.short_term = self.short_term[-20:]
```

---

## 8. Agent 评估最佳实践

### 8.1 设计评估任务

```python
# 评估任务: 数据分析 Agent
tasks = [
    {
        "input": "分析 sales.csv, 计算每月销售额",
        "expected_actions": ["read_file", "run_python", "calculate"],
        "expected_output": "包含每月销售额的表格"
    },
    {
        "input": "找出销售最好的产品类别",
        "expected_actions": ["read_file", "run_python", "aggregate"],
        "expected_output": "类别名 + 销售额"
    }
]

# 评估
for task in tasks:
    output, actions = agent.run(task['input'])
    
    # 检查动作序列
    action_match = set(task['expected_actions']) <= set(actions)
    
    # 检查输出
    output_correct = check_output(output, task['expected_output'])
    
    print(f"任务: {task['input'][:30]} | 动作匹配: {action_match} | 输出正确: {output_correct}")
```

### 8.2 错误分析

```python
# 收集失败案例
failures = []
for task in test_tasks:
    try:
        output = agent.run(task['input'])
        if not is_correct(output, task['expected']):
            failures.append({
                "task": task,
                "output": output,
                "expected": task['expected'],
                "trace": agent.last_trace
            })
    except Exception as e:
        failures.append({
            "task": task,
            "error": str(e),
            "trace": agent.last_trace
        })

# 分析
error_types = Counter()
for f in failures:
    error_type = classify_error(f)
    error_types[error_type] += 1

print("错误类型分布:", error_types)
# Counter({'工具选择错误': 5, '参数错误': 3, '超时': 2, ...})
```

---

## 9. Agent 开发框架对比

| 框架 | 特点 | 学习曲线 | 适用 |
| :--- | :--- | :--- | :--- |
| **LangChain** | 工具链齐全 | 中 | 通用 |
| **LlamaIndex** | RAG 强 | 低 | 数据 Agent |
| **CrewAI** | 多 Agent 易用 | 低 | 协作场景 |
| **AutoGen** | 对话 Agent | 中 | 复杂对话 |
| **LangGraph** | 图结构 | 中高 | 状态机 |
| **OpenAI Swarm** | 极简 | 低 | 轻量级 |
| **AgentScope** | 全栈 | 中 | 生产级 |

---

## 10. 实战: Web 搜索 + 总结 Agent

```python
from openai import OpenAI
import requests

class WebSearchAgent:
    def __init__(self, api_key, search_api_key):
        self.llm = OpenAI(api_key=api_key)
        self.search_api_key = search_api_key
    
    def search_web(self, query):
        """调用搜索 API (如 Serper, Bing)"""
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": self.search_api_key}
        data = {"q": query}
        response = requests.post(url, headers=headers, json=data)
        return response.json()
    
    def run(self, query):
        messages = [
            {"role": "system", "content": "你是研究助手, 使用 web_search 工具搜索信息, 然后总结回答。"},
            {"role": "user", "content": query}
        ]
        
        for step in range(5):
            response = self.llm.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "搜索网页",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"]
                        }
                    }
                }]
            )
            
            msg = response.choices[0].message
            messages.append(msg)
            
            if not msg.tool_calls:
                return msg.content
            
            for tc in msg.tool_calls:
                if tc.function.name == "web_search":
                    args = json.loads(tc.function.arguments)
                    results = self.search_web(args['query'])
                    
                    # 简化结果
                    summary = "\n".join([
                        f"- {r.get('title', '')}: {r.get('snippet', '')}"
                        for r in results.get('organic', [])[:5]
                    ])
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": summary
                    })
        
        return "未完成"

# 使用
agent = WebSearchAgent(api_key="...", search_api_key="...")
result = agent.run("2025 年 LLM 最新进展是什么?")
print(result)
```

---

## 11. 关键公式汇总

### 11.1 ReAct

$$\text{Thought}_t, \text{Action}_t, \text{Observation}_t = f_{LLM}(\text{History}_{t-1})$$
$$\text{History}_t = \text{History}_{t-1} + (\text{Thought}_t, \text{Action}_t, \text{Observation}_t)$$

### 11.2 Agent 决策概率

$$P(a_t | s_t) = f_\pi(s_t)$$
其中 $s_t$ 是当前状态 (历史 + 工具结果), $a_t$ 是动作.

### 11.3 任务规划

$$P(\text{plan} | \text{goal}) = f_{LLM}(\text{goal})$$
$$\text{plan} = [\tau_1, \tau_2, \ldots, \tau_n]$$
$$\text{success} = \prod_{i=1}^n P(\tau_i \text{ succeeds})$$

---

## 12. 推荐学习路径

1. **基础** (1 周): 读 ReAct 论文, 理解 Agent 循环
2. **简单实现** (1 周): 用本附录代码搭出 ReAct Agent
3. **工具使用** (1 周): 学习 Function Calling, 集成多个工具
4. **记忆系统** (1 周): 实现短期/长期记忆
5. **规划能力** (1 周): 实现 Plan-and-Execute
6. **多 Agent** (1 周): CrewAI/AutoGen 实现协作

---

## 13. 参考资源

- [datawhalechina/hello-agents](https://github.com/datawhalechina/hello-agents) — 从零开始构建智能体
- [datawhalechina/agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab) — Agentic RL 实验室
- [LangChain](https://python.langchain.com/) — Agent 框架
- [CrewAI](https://www.crewai.com/) — 多 Agent 框架
- [AutoGen](https://github.com/microsoft/autogen) — 微软多 Agent

### 经典论文
1. **ReAct**: Yao et al., 2022. *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023.
2. **Reflexion**: Shinn et al., 2023. *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS.
3. **Toolformer**: Schick et al., 2023. *Toolformer: Language Models Can Teach Themselves to Use Tools*. arXiv:2302.04761.
4. **HuggingGPT**: Shen et al., 2023. *HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace*. arXiv:2303.17580.

---

> 本附录覆盖 Agent 的核心原理与实战. 配合 [算法与AI/Agent技术/](./) 整个目录阅读, 可以全面掌握 Agent 开发.
