import autogen
from autogen.agentchat import AssistantAgent
from config import LLM_CONFIG
import json
from fix_busted_json import repair_json

CRITIC_AGENT_NAME = "PaperSearchCriticAgent"

critic_llm_config = LLM_CONFIG.copy()

critic_agent = AssistantAgent(
    name=CRITIC_AGENT_NAME,
    llm_config=critic_llm_config,
    system_message=f"""You are an AI Critic Agent. Your role is to evaluate the performance of another AI agent,
the 'PaperSearchAssistant', which is designed to find research papers using a tool.

You will be given:
1. The original 'User Prompt' given to the PaperSearchAssistant.
2. The 'Agent's Final Response' (the last message from PaperSearchAssistant intended for the user).
3. The full 'Conversation History' between the User Proxy and the PaperSearchAssistant.

Based on this information, evaluate the PaperSearchAssistant's performance according to the following criteria,
providing a score from 1 (Poor) to 5 (Excellent) for each:

1.  **Completeness (1-5):** Did the agent fully address every aspect of the user's prompt (topic, year constraints, citation constraints, number of papers requested)?
    Consider if all parts of the request were covered in the final response or if the agent acknowledged limitations.
2.  **Quality/Accuracy (1-5):**
    - Was the response accurate (did it find relevant papers that actually match the criteria mentioned in the prompt)? You'll need to infer this from the agent's response; you don't have ground truth access to a database. Assume the agent's reported paper details (title, year, citations) are true *if the tool was called*.
    - If the tool was NOT called and the agent hallucinated papers, score this very low (1).
    - Was the information presented clearly, well-organized, and easy to understand?
3.  **Robustness (1-5):**
    - How well did the agent handle ambiguous prompts? Did it ask good, relevant clarifying questions if needed (check conversation history)?
    - How did it handle edge cases or impossible requests? Did it use its tool appropriately and report 'no results' or 'error' sensibly, or did it fail/hallucinate?
4.  **Tool Usage (1-5):** (Refer to conversation history for actual tool calls if any)
    - Did the agent correctly identify when to use its 'search_research_papers' tool?
    - Did it extract parameters for the tool reasonably correctly from the user's prompt? (e.g., topic, year, year_filter, min_citations, limit).
    - Did it correctly interpret the tool's output (both success and error messages)? Check if its summary matches what the tool provided.
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
Be objective and fair.
"""
)

def evaluate_agent_response(user_prompt: str, agent_final_response: str, conversation_history: list) -> dict:
    """
    Uses the LLM Critic agent to evaluate the paper search agent's response.
    """
    # Format the conversation history for the critic prompt
    formatted_history = []
    for msg in conversation_history:
        role = msg.get("role", "unknown")
        name = msg.get("name", role) # Use name if available, else role
        content = msg.get("content", "")
        # Check for tool calls and responses in content if it's a dict
        if isinstance(content, list) and content and "tool_calls" in content[0]: # OpenAI format tool call
             content_str = f"Tool Call Suggested: {json.dumps(content[0]['tool_calls'])}"
        elif isinstance(content, str) and content.startswith("***** Response from calling tool"): # Our user_proxy format
             content_str = content # Keep as is
        elif isinstance(content, dict) and "tool_responses" in content: # OpenAI format tool response
             content_str = f"Tool Response: {json.dumps(content['tool_responses'])}"
        else:
            content_str = str(content) # Ensure content is string
        formatted_history.append(f"From {name} ({role}):\n{content_str}\n-------")
    history_str = "\n".join(formatted_history)

    # Ensure agent_final_response is a string
    if not isinstance(agent_final_response, str):
        agent_final_response = str(agent_final_response)


    critic_request_prompt = f"""
User Prompt to PaperSearchAssistant:
------------------------------------
{user_prompt}
------------------------------------

PaperSearchAssistant's Final Response to User:
---------------------------------------------
{agent_final_response}
---------------------------------------------

Full Conversation History (UserQueryProxy and PaperSearchAssistant):
------------------------------------------------------------------
{history_str}
------------------------------------------------------------------

Please provide your evaluation as a JSON object based on the criteria outlined in your system message.
"""

    print(f"\n🔍 Critic evaluating response for prompt: '{user_prompt[:50]}...'")
    # The critic_agent.generate_reply expects a list of messages
    critic_response_message = critic_agent.generate_reply(
        messages=[{"role": "user", "content": critic_request_prompt}]
    )

    # Extract the content from the message object
    # In newer Autogen, generate_reply returns a message object or a string.
    # We need to handle both cases.
    if isinstance(critic_response_message, dict) and "content" in critic_response_message:
        critic_evaluation_str = critic_response_message["content"]
    elif isinstance(critic_response_message, str):
        critic_evaluation_str = critic_evaluation_str
    else:
         print("Warning: Critic response was not in expected format (dict with 'content' or str).")
         print(f"RAW CRITIC RESPONSE: {critic_response_message}")
         return {"error": "Critic returned unexpected response format", "raw_response": str(critic_response_message)}

    print(f"📝 Critic raw response (full): >>>\n{critic_evaluation_str}\n<<<")

    text_to_parse = critic_evaluation_str.strip()
    if text_to_parse.startswith("```json"):
        text_to_parse = text_to_parse[len("```json"):].strip()
        if text_to_parse.endswith("```"):
            text_to_parse = text_to_parse[:-len("```")].strip()
    elif text_to_parse.startswith("```"):
        text_to_parse = text_to_parse[len("```"):].strip()
        if text_to_parse.endswith("```"):
            text_to_parse = text_to_parse[:-len("```")].strip()

    print(f"🧐 Text after stripping markdown: >>>\n{text_to_parse}\n<<<")

    try:
        print(f"🧐 String being passed to fix_json: >>>\n{text_to_parse}\n<<<")
        fixed_json_str = repair_json(text_to_parse)
        print(f"🔧 String after fix_json: >>>\n{fixed_json_str}\n<<<")

        evaluation_json = json.loads(fixed_json_str)
        return evaluation_json

    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON even after fix_json: {e}")
        print(f"   Problematic string (after fix_json) was: >>>\n{fixed_json_str}\n<<<")
        return {
            "error": "Failed to decode JSON from critic even after fixing",
            "original_raw_response": critic_evaluation_str,
            "markdown_stripped_response": text_to_parse,
            "fixed_attempt_response": fixed_json_str,
        }
    except Exception as e:
        print(f"❌ An unexpected error occurred during JSON fixing or parsing: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": "Unexpected error during JSON fixing/parsing",
            "original_raw_response": critic_evaluation_str,
        }

# --- Basic Test for the Critic (Optional) ---
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
        # Simulate a tool call and response for testing robustness of critic
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