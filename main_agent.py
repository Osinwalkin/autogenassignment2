import autogen
from autogen.agentchat import AssistantAgent, UserProxyAgent
from config import LLM_CONFIG
from research_tools import search_research_papers, search_research_papers_tool_schema
import json

ASSISTANT_AGENT_NAME = "PaperSearchAssistant"
USER_PROXY_AGENT_NAME = "UserQueryProxy"
MAX_CONSECUTIVE_AUTO_REPLY = 8

ASSISTANT_SYSTEM_MESSAGE = f"""You are a helpful AI assistant specialized in finding research papers.
You have access to a function 'search_research_papers' to search Semantic Scholar.
The schema for this function is: {json.dumps(search_research_papers_tool_schema, indent=2)}

Your goal is to fulfill the user's request for research papers.
When a user asks for research papers, carefully analyze their request to extract:
- The 'topic' (required).
- An optional 'year' and 'year_filter' (e.g., 'in', 'before', 'after').
- An optional 'min_citations'.
- An optional 'limit' on the number of results (default to 5 if not specified by the user, but use your judgment if more are implicitly requested, up to a maximum of 10 unless specified higher).

**Interaction Flow:**
1.  If the user's request is clear and provides all necessary information (at least 'topic'), proceed to call the 'search_research_papers' function.
2.  If crucial information like 'topic' is missing, or if terms like 'recent' are used without specific years, you MUST ask clarifying questions.
3.  If you ask a clarifying question and the user (UserQueryProxy) provides an empty or unhelpful response, state that you cannot proceed without the necessary clarification and then reply TERMINATE. Do NOT proceed with default assumptions if clarification was sought but not adequately provided.
4.  When calling the 'search_research_papers' function, ensure you provide arguments matching the schema.
5.  Do not make up paper details or sources. Only provide information returned by the function.

**Presenting Results:**
- After receiving the JSON string result from the function:
    - If it's an error message from the tool (e.g., "No papers found matching your criteria.", "HTTP error occurred"), inform the user clearly about this outcome.
    - If it's a list of papers, present the information neatly. For each paper, clearly state: Title, Authors, Year, Citation Count. Also include the URL and DOI if available.
    - Do not just dump the raw JSON to the user.

**Termination:**
- Once you have successfully presented the papers, Reply TERMINATE.
- Do not ask "Is there anything else?" or similar follow-up questions after the task is complete.
"""

def create_paper_search_agents() -> tuple[UserProxyAgent, AssistantAgent]:
    """Initializes and returns the UserProxyAgent and AssistantAgent."""
    
    assistant_llm_config_tools = {
        **LLM_CONFIG,
        "tools": [
            {
                "type": "function",
                "function": search_research_papers_tool_schema,
            }
        ],
    }
    
    assistant = AssistantAgent(
        name=ASSISTANT_AGENT_NAME,
        llm_config=assistant_llm_config_tools,
        system_message=ASSISTANT_SYSTEM_MESSAGE,
        function_map={
            search_research_papers_tool_schema["name"]: search_research_papers
        }
    )
    
    user_proxy = UserProxyAgent(
        name=USER_PROXY_AGENT_NAME,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=MAX_CONSECUTIVE_AUTO_REPLY,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config=False,
        function_map={
            search_research_papers_tool_schema["name"]: search_research_papers
        }
    )
    return user_proxy, assistant

def run_paper_search_chat(task_message: str, user_proxy: UserProxyAgent, assistant: AssistantAgent) -> tuple[str, list]:
    """
    Runs a chat between the provided UserProxyAgent and AssistantAgent for a given task.
    Resets agents before chat.

    Args:
        task_message (str): The user's request/prompt.
        user_proxy (UserProxyAgent): The user proxy agent.
        assistant (AssistantAgent): The assistant agent.

    Returns:
        tuple: (agent_final_user_facing_response_str, full_conversation_history_list)
    """
    # Reset agents to clear previous state/history for this specific chat
    user_proxy.reset()
    assistant.reset()

    print(f"💬 {user_proxy.name} initiating chat with {assistant.name} for task: '{task_message}'")
    user_proxy.initiate_chat(
        recipient=assistant,
        message=task_message,
    )
    print(f"🏁 Chat completed for task: '{task_message}'")

    # --- Extracting Final Response and History ---
    agent_final_user_facing_response = "No suitable user-facing response found from assistant."

    full_conversation_history = user_proxy.chat_messages.get(assistant, [])

    if not full_conversation_history:
        print(f"Warning: No chat history found in {user_proxy.name} for {assistant.name}.")
        return agent_final_user_facing_response, []

    # Find the last user-facing message from the ASSISTANT
    for msg in reversed(full_conversation_history):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            
            is_tool_call_generated = False
            if isinstance(content, str):
                try:
                    if "tool_calls" in content or "function_call" in content:
                        potential_tool_call_data = json.loads(content)
                        if isinstance(potential_tool_call_data, dict) and \
                           (potential_tool_call_data.get("tool_calls") or potential_tool_call_data.get("function_call")):
                            is_tool_call_generated = True
                except json.JSONDecodeError:
                    pass 
            elif isinstance(content, list) and content and isinstance(content[0], dict) and "tool_calls" in content[0]:
                 is_tool_call_generated = True 
            
            if not is_tool_call_generated and isinstance(content, str) and content.strip():
                agent_final_user_facing_response = content
                break 

    return agent_final_user_facing_response, full_conversation_history


# Module for testing main_agent.py
if __name__ == "__main__":
    print("--- Testing Paper Search Agent Module ---")
    
    user_proxy_agent, assistant_agent = create_paper_search_agents()

    test_prompts_for_module = [
        "Find 3 research papers on 'transformer models in NLP' published in 2021 with more than 200 citations.",
        #"I need some recent papers on reinforcement learning.",
        #"Search for papers on '' with 10 citations."
    ]

    for i, task in enumerate(test_prompts_for_module):
        print(f"\n--- Module Test Case {i+1} ---")
        print(f"User Task: {task}")
        
        final_response, history = run_paper_search_chat(task, user_proxy_agent, assistant_agent)
        
        print(f"\nAgent's Final User-Facing Response:\n{final_response}")
    
    print("\n--- Paper Search Agent Module Testing Finished ---")