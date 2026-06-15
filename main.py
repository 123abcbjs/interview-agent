import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, StateGraph


ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT / "examples"
REPORTS_DIR = ROOT / "reports"
TOTAL_ROUNDS = 3


class InterviewState(TypedDict):
    jd: str
    resume: str
    analysis: str            # 保存岗位和简历的分析
    current_focus: str       # 保存下一轮的考察重点
    next_action: str         # 表示追问or切换方向
    round_number: int        # 3轮后终止
    transcript: list[dict[str, str]] # 保存完整的问答记录和评价记录
    report: str


def message_text(response) -> str:
    content = response.content
    if isinstance(content, str):
        return content.strip()
    return "\n".join(
        part.get("text", "") for part in content if isinstance(part, dict)
    ).strip()


def ask_model(model: ChatDeepSeek, prompt: str) -> str:
    try:
        return message_text(model.invoke(prompt))
    except Exception as exc:
        raise RuntimeError(f"DeepSeek 调用失败：{exc}") from exc


def relevant_lines(text: str, query: str, limit: int = 4) -> str:
    """用关键词检索最相关的几行信息"""
    keywords = {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}|[\u4e00-\u9fff]{2,4}", query)
    }
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    ranked = sorted(
        lines,
        key=lambda line: sum(word in line.lower() for word in keywords),
        reverse=True,
    )
    selected = ranked[:limit]
    return "\n".join(selected) if selected else text[:800]


def build_graph(model: ChatDeepSeek):
    def analyze_materials(state: InterviewState):
        prompt = f"""你是一名法律岗位招聘助手。请阅读岗位 JD 和候选人简历，
            从法律基础、实务能力、法律检索与文书写作、职业伦理与风险意识中，
            提取 3 个最值得考察的方向，并指出简历中需要核实的内容。
            不要虚构候选人经历或法律规定。
            回答控制在 180 字以内。

            岗位 JD：
            {state['jd']}

            候选人简历：
            {state['resume']}
            """
        analysis = ask_model(model, prompt)
        return {
            "analysis": analysis,
            "current_focus": "法律基础、实务能力与职业判断",
            "next_action": "NEXT",
        }

    def generate_question(state: InterviewState):
        round_number = state["round_number"] + 1
        previous = state["transcript"][-1] if state["transcript"] else None
        query = state["current_focus"]
        jd_context = relevant_lines(state["jd"], query)
        resume_context = relevant_lines(state["resume"], query)

        previous_context = "这是第一题。"
        if previous:
            previous_context = (
                f"上一题：{previous['question']}\n"
                f"候选人回答：{previous['answer']}\n"
                f"评价：{previous['evaluation']}\n"
                f"下一步：{state['next_action']}"
            )

        prompt = f"""你是一名友好但认真的法律岗位面试官，现在进行第 {round_number} 轮面试。
            只生成一个简洁、适合法学本科生应聘法务实习岗位的问题。
            如果下一步是 FOLLOW_UP，请针对上一轮回答中不清楚的地方追问；
            否则切换到尚未考察的重要方向。问题应考察法律分析、事实与证据意识、
            风险判断或法律文书能力。不要要求披露客户或案件机密，不要虚构案情、
            法条编号或候选人经历。不要输出答案或解释。

            材料分析：
            {state['analysis']}

            相关 JD：
            {jd_context}

            相关简历：
            {resume_context}

            上一轮信息：
            {previous_context}
            """
        question = ask_model(model, prompt)
        print(f"\n第 {round_number}/{TOTAL_ROUNDS} 题：{question}")
        answer = input("你的回答：").strip()
        if not answer:
            answer = "候选人未作答。"

        transcript = list(state["transcript"])
        transcript.append(
            {"question": question, "answer": answer, "evaluation": ""}
        )
        return {"round_number": round_number, "transcript": transcript}

    def evaluate_answer(state: InterviewState):
        item = state["transcript"][-1]
        prompt = f"""你是一名法律岗位面试评审。请评价候选人的回答。
            第一行必须仅输出 DECISION: FOLLOW_UP 或 DECISION: NEXT。
            第二行输出 FOCUS: 后续重点。
            之后用不超过 100 字说明回答中的有效证据、不足和建议。
            重点判断候选人能否识别法律问题、说明适用规则或检索路径、结合事实与证据
            进行推理，并识别风险。没有依据的确定性法律结论不能视为有效回答。
            只有回答含糊、缺少关键分析过程时才选择 FOLLOW_UP。

            岗位与简历分析：
            {state['analysis']}

            问题：{item['question']}
            回答：{item['answer']}
            """
        evaluation = ask_model(model, prompt)
        decision = "FOLLOW_UP" if "DECISION: FOLLOW_UP" in evaluation.upper() else "NEXT"
        focus_match = re.search(r"FOCUS:\s*(.+)", evaluation, re.IGNORECASE)
        focus = focus_match.group(1).strip() if focus_match else "下一个法律岗位核心能力"

        transcript = list(state["transcript"])
        transcript[-1] = {**item, "evaluation": evaluation}
        print(f"\n简要评价：\n{evaluation}")
        return {
            "transcript": transcript,
            "next_action": decision,
            "current_focus": focus,
        }

    def generate_report(state: InterviewState):
        transcript_text = "\n\n".join(
            f"第 {index} 题：{item['question']}\n"
            f"回答：{item['answer']}\n"
            f"评价：{item['evaluation']}"
            for index, item in enumerate(state["transcript"], start=1)
        )
        prompt = f"""请根据以下三轮法律岗位面试生成一份简洁的 Markdown 面试报告。
            必须包含：总体评价、回答证据、法律基础与实务判断、风险意识、优势、
            待提升项、后续学习建议。
            评价必须引用候选人的实际回答，不要虚构经历、法律规定或法条编号。
            本报告仅用于模拟面试辅助，不构成法律意见。

            岗位与简历分析：
            {state['analysis']}

            面试记录：
            {transcript_text}
            """
        return {"report": ask_model(model, prompt)}

    graph = StateGraph(InterviewState)
    graph.add_node("analyze", analyze_materials)
    graph.add_node("question", generate_question)
    graph.add_node("evaluate", evaluate_answer)
    graph.add_node("report", generate_report)
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "question")
    graph.add_edge("question", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        lambda state: "report" if state["round_number"] >= TOTAL_ROUNDS else "question",
        {"question": "question", "report": "report"},
    )
    graph.add_edge("report", END)
    return graph.compile()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"找不到文件：{path}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"文件不是 UTF-8 编码：{path}") from exc


def parse_args():
    parser = argparse.ArgumentParser(description="基于 LangGraph 的自适应法律岗位面试 Agent")
    parser.add_argument("--jd", type=Path, default=EXAMPLES_DIR / "jd.txt")
    parser.add_argument("--resume", type=Path, default=EXAMPLES_DIR / "resume.txt")
    return parser.parse_args()


def save_report(report: str) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"interview_{timestamp}.md"
    path.write_text(report.strip() + "\n", encoding="utf-8")
    return path


def main() -> int:
    load_dotenv(ROOT / ".env")
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误：未找到 DEEPSEEK_API_KEY。请配置环境变量或 interview_agent/.env。")
        return 1

    args = parse_args()
    try:
        jd = read_text(args.jd)
        resume = read_text(args.resume)
        model = ChatDeepSeek(model="deepseek-chat", temperature=0.3, max_tokens=800)
        app = build_graph(model)
        result = app.invoke(
            {
                "jd": jd,
                "resume": resume,
                "analysis": "",
                "current_focus": "",
                "next_action": "NEXT",
                "round_number": 0,
                "transcript": [],
                "report": "",
            }
        )
        report_path = save_report(result["report"])
    except (ValueError, RuntimeError) as exc:
        print(f"错误：{exc}")
        return 1
    except KeyboardInterrupt:
        print("\n面试已取消。")
        return 130

    print("\n===== 面试报告 =====\n")
    print(result["report"])
    print(f"\n报告已保存至：{report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
