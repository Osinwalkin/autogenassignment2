import autogen
from autogen.agentchat import AssistantAgent
from config import LLM_CONFIG
import json
from fix_busted_json import repair_json
import traceback
import re

CRITIC_AGENT_NAME = "PaperSearchCriticAgent"

CRITIC_SYSTEM_MESSAGE = f"""You are an AI Critic Agent. Your role is to evaluate the performance of another AI agent,
the 'PaperSearchAssistant', which is designed to find research papers using a tool.

You will be given:
1. The original 'User Prompt' given to the PaperSearchAssistant.
2. The 'Agent's Final Response' (the last user-facing message from PaperSearchAssistant).
3. The full 'Conversation History' between the User Proxy and the PaperSearchAssistant.

Based on this information, evaluate the PaperSearchAssistant's performance according to the following criteria,
providing a score from 1 (Poor) to 5 (Excellent) for each:

1.  **Completeness (1-5):** Did the agent fully address every aspect of the user's prompt (topic, year constraints, citation constraints, number of papers requested)?
    Consider if all parts of the request were covered in the final response or if the agent acknowledged limitations.
2.  **Quality/Accuracy (1-5):**
    - Was the response accurate (did it find relevant papers that actually match the criteria mentioned in the prompt)? You'll need to infer this from the agent's response; you don't have ground truth access to a database. Assume the agent's reported paper details (title, year, citations) are true *if the tool was called and the response seems to reflect tool output*.
    - If the tool was NOT called when it should have been and the agent hallucinated papers, score this very low (1).
    - Was the information presented clearly, well-organized, and easy to understand?
3.  **Robustness (1-5):**
    - How well did the agent handle ambiguous prompts? Did it ask good, relevant clarifying questions if needed (check conversation history)?
    - How did it handle edge cases or impossible requests? Did it use its tool appropriately and report 'no results' or 'error' sensibly, or did it fail/hallucinate?
4.  **Tool Usage (1-5):** (Refer to conversation history for actual tool calls and tool responses if any)
    - Did the agent correctly identify when to use its 'search_research_papers' tool?
    - Did it extract parameters for the tool reasonably correctly from the user's prompt?
    - Did it correctly interpret the tool's output (both success and error messages from the tool)? Check if its summary aligns with evidence of tool output in the history.
    - If no tool call was made when one was clearly appropriate, score this low. If a tool call was made unnecessarily, also score lower.
5.  **Efficiency/Conciseness (1-5):**
    - Did the agent achieve the task with a reasonable number of conversational turns (check conversation history)? Too much back-and-forth for simple queries is inefficient.
    - Was the final answer concise and to the point, without unnecessary chatter after fulfilling the request?

Provide your evaluation as a JSON object with the following fields:
- "completeness_score": integer (1-5)
- "quality_accuracy_score": integer (1-5)
- "robustness_score": integer (1-5)
- "tool_usage_score": integer (1-5)
- "efficiency_conciseness_score": integer (1-5)
- "overall_assessment": (string) A brief overall summary of the agent's performance on this specific task.
- "positive_feedback": (string) Specific positive aspects, with examples from the response or history.
- "areas_for_improvement": (string) Specific areas where the agent could improve, with examples. Mention if the tool was not called but should have been, or if it hallucinated.

Focus your evaluation solely on the provided prompt, response, and history. Do not make assumptions beyond this data.
Be objective and fair. Ensure your output is a single, valid JSON object.
"""

critic_llm_config = LLM_CONFIG.copy()

critic_agent = AssistantAgent(
    name=CRITIC_AGENT_NAME,
    llm_config=critic_llm_config,
    system_message=CRITIC_SYSTEM_MESSAGE
)


# Formats the conversation history for inclusion in the critic's prompt.
def format_history_for_critic(conversation_history: list) -> str:
    formatted_messages = []
    for msg in conversation_history:
        role = msg.get("role", "unknown_role")
        
        sender_name = msg.get("name")
        if not sender_name or sender_name == role:
            sender_display = role.upper()
        else:
            sender_display = f"{sender_name} ({role})"

        content = msg.get("content", "")
        content_str = ""

        if isinstance(content, str):
            if content.startswith("***** Response from calling tool"):
                content_str = f"TOOL_RESPONSE:\n{content}"
            else:
                content_str = content
        elif isinstance(content, list) and content and isinstance(content[0], dict) and "tool_calls" in content[0]:
            content_str = f"ASSISTANT_SUGGESTS_TOOL_CALL: {json.dumps(content[0]['tool_calls'])}"

        else:
            content_str = str(content)

        content_str_oneline = content_str.replace("\n", " <NEWLINE> ")
        formatted_messages.append(f"FROM {sender_display}:\n{content_str_oneline}\n-----------------------------")
    
    return "\n".join(formatted_messages) if formatted_messages else "No conversation history provided."


# Uses the LLM Critic agent to evaluate the paper search agent's response.
def evaluate_agent_response(user_prompt: str, agent_final_response: str, conversation_history: list) -> dict:
    history_str = format_history_for_critic(conversation_history)

    if not isinstance(agent_final_response, str):
        agent_final_response = str(agent_final_response)

    critic_request_prompt = f"""User Prompt to PaperSearchAssistant:
====================================
{user_prompt}
====================================

PaperSearchAssistant's Final User-Facing Response:
================================================
{agent_final_response}
================================================

Full Conversation History (between UserQueryProxy and PaperSearchAssistant):
==========================================================================
{history_str}
==========================================================================

Please provide your evaluation as a single, valid JSON object based on the criteria outlined in your system message.
Do not include any explanatory text before or after the JSON object itself.
"""

    print(f"\n🔍 Critic evaluating response for prompt: '{user_prompt[:50]}...'")
    critic_response_message = critic_agent.generate_reply(
        messages=[{"role": "user", "content": critic_request_prompt}]
    )

    critic_evaluation_str = ""
    if isinstance(critic_response_message, dict) and "content" in critic_response_message:
        critic_evaluation_str = critic_response_message["content"]
    elif isinstance(critic_response_message, str):
        critic_evaluation_str = critic_response_message
    else:
         print(f"Warning: Critic response was not in expected format. RAW CRITIC RESPONSE: {critic_response_message}")
         return {"error": "Critic returned unexpected response format", "raw_response": str(critic_response_message)}

    print(f" Critic raw response (full): >>>\n{critic_evaluation_str}\n<<<")

    text_to_parse = critic_evaluation_str.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", critic_evaluation_str, re.IGNORECASE)
    if match:
        text_to_parse = match.group(1).strip()
    else:
        text_to_parse = critic_evaluation_str.strip()
    
    print(f" Text after stripping markdown: >>>\n{text_to_parse}\n<<<")

    repaired_json_str = ""
    try:
        print(f" String being passed to repair_json: >>>\n{text_to_parse}\n<<<")
        repaired_json_str = repair_json(text_to_parse)
        print(f" String after repair_json: >>>\n{repaired_json_str}\n<<<")

        evaluation_json = json.loads(repaired_json_str)
        return evaluation_json

    except json.JSONDecodeError as e:
        print(f" Error decoding JSON even after repair_json: {e}")
        print(f"   Problematic string (after repair_json) was: >>>\n{repaired_json_str}\n<<<")
        return {
            "error": "Failed to decode JSON from critic even after repairing",
            "original_raw_response": critic_evaluation_str,
            "markdown_stripped_response": text_to_parse,
            "repaired_attempt_response": repaired_json_str,
        }
    except Exception as e:
        print(f" An unexpected error occurred during JSON repairing or parsing: {e}")
        traceback.print_exc()
        return {
            "error": "Unexpected error during JSON repairing/parsing",
            "original_raw_response": critic_evaluation_str,
            "details": str(e)
        }

# Tests for evaluation.py
if __name__ == "__main__":
    print("--- Testing Critic Agent ---")
    sample_user_prompt = "Find me 2 papers on 'AI ethics' published after 2022."
    sample_agent_response = """
Here are two papers on AI ethics published after 2022:
1. Title: The Moral Machine Experiment, Year: 2023, Citations: 150, URL: example.com/moralmachine
2. Title: Algorithmic Bias and Fairness, Year: 2024, Citations: 90, URL: example.com/bias
TERMINATE
"""
    sample_history = [
        {"role": "user", "name": "UserQueryProxy", "content": sample_user_prompt},
        {"role": "assistant", "name": "PaperSearchAssistant", "content": "Okay, I will search for that."},
        {"role": "assistant", "name": "PaperSearchAssistant", "content": json.dumps([{"tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "search_research_papers", "arguments": json.dumps({"topic": "AI ethics", "year": 2022, "year_filter": "after", "limit": 2})}}]}])},
        {"role": "user", "name": "UserQueryProxy", "content": json.dumps([{"tool_call_id": "call_123", "role": "tool", "name": "search_research_papers", "content": json.dumps([{"title": "The Moral Machine Experiment", "year": 2023, "citationCount": 150, "url": "example.com/moralmachine"}, {"title": "Algorithmic Bias and Fairness", "year": 2024, "citationCount": 90, "url": "example.com/bias"}])}])},
        {"role": "assistant", "name": "PaperSearchAssistant", "content": sample_agent_response}
    ]

    evaluation = evaluate_agent_response(sample_user_prompt, sample_agent_response, sample_history)
    print("\nCritic's Evaluation:")
    print(json.dumps(evaluation, indent=2))

    # Test with a hallucinated response (no tool call)
    print("\n--- Testing Critic with Hallucinated Response ---")
    hallucinated_agent_response = "I found a great paper: 'Future of AI' by Dr. Who, 2077. Highly cited! TERMINATE"
    hallucinated_history = [
        {"role": "user", "name": "UserQueryProxy", "content": sample_user_prompt},
        {"role": "assistant", "name": "PaperSearchAssistant", "content": "Let me check..."},
        {"role": "assistant", "name": "PaperSearchAssistant", "content": hallucinated_agent_response} # No tool call evidence
    ]
    evaluation_hallucinated = evaluate_agent_response(sample_user_prompt, hallucinated_agent_response, hallucinated_history)
    print("\nCritic's Evaluation (Hallucinated):")
    print(json.dumps(evaluation_hallucinated, indent=2))