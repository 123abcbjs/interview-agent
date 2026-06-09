# 自适应技术面试 Agent

这是一个用于学习 LangGraph 的简单命令行项目。程序读取岗位 JD 和候选人简历，
进行 3 轮技术面试，并根据上一轮回答决定继续追问还是切换方向。面试结束后会生成
一份 Markdown 报告。

## 功能

- 使用 LangGraph 组织分析、提问、评价和报告生成流程
- 使用 DeepSeek 生成问题并评价回答
- 使用简单关键词匹配选择相关 JD 和简历片段
- 支持替换自定义 JD 与简历文本文件
- 将面试结果保存为 Markdown

## 安装

```powershell
cd C:\Users\26873\Documents\简历求职\interview_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

配置 DeepSeek API Key：

```powershell
$env:DEEPSEEK_API_KEY="你的 API Key"
```

也可以在项目目录创建 `.env`：

```text
DEEPSEEK_API_KEY=你的 API Key
```

## 运行

使用内置示例：

```powershell
python main.py
```

使用自定义 UTF-8 文本文件：

```powershell
python main.py --jd .\my_jd.txt --resume .\my_resume.txt
```

报告会保存到 `reports/` 目录。

## 工作流

```text
分析 JD 与简历 -> 生成问题 -> 用户回答 -> 评价回答
                                      |
                          未完成 3 轮则继续提问
                                      |
                                  生成报告
```

## 项目局限

- 当前仅进行 3 轮面试，没有网页界面和用户系统。
- “检索”采用关键词匹配，没有使用向量数据库。
- 这是单个 LangGraph 工作流，不是真正的多 Agent 系统。
- 评价结果由大模型生成，只适合作为辅助参考。
