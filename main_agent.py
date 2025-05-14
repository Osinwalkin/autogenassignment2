import autogen
from autogen.agentchat import AssistantAgent, UserProxyAgent
from config import LLM_CONFIG
from research_tools import search_research_papers, search_research_papers_tool_schema

def run_paper_search_chat(task_message: str) -> tuple[str, list]:
    """
    Runs a chat with the PaperSearchAssistant for a given task.

    Args:
        task_message (str): The user's request/prompt.

    Returns:
        tuple: (agent_final_response_str, conversation_history_list)
    """
    print(f"\n🚀 Initializing agents for task: {task_message[:60]}...")
    assistant_llm_config = {
        **LLM_CONFIG,
        "tools": [
            {
                "type": "function",
                "function": search_research_papers_tool_schema,
            }
        ],
    }
    assistant = AssistantAgent(
        name="PaperSearchAssistant",
        llm_config=assistant_llm_config,
        system_message="""You are a helpful AI assistant specialized in finding research papers.
    You have access to a function 'search_research_papers' to search Semantic Scholar.
    Its schema is: {search_research_papers_tool_schema}

    When a user asks for research papers, analyze their request to extract:
    - The 'topic' (required).
    - An optional 'year' and 'year_filter' (in, before, after).
    - An optional 'min_citations'.
    - An optional 'limit' on the number of results (default to 5 if not specified).

    If any required information for the tool is missing or ambiguous, ask clarifying questions.
    When you have enough information, call the 'search_research_papers' function.
    Do not make up answers or paper details. Only provide information returned by the function.

    After receiving the JSON string result from the function:
    - If it's an error message, inform the user about the error.
    - If it's a list of papers, present the information clearly. For each paper, mention: Title, Authors, Year, Citation Count, and URL or DOI.
    - If the tool returns a "No papers found" message, inform the user.

    Reply TERMINATE when the task is fully complete.
    """,
        function_map={
            search_research_papers_tool_schema["name"]: search_research_papers
        }
    )
    user_proxy = UserProxyAgent(
        name="UserQueryProxy",
        human_input_mode="NEVER", # Important for automated evaluation
        max_consecutive_auto_reply=8, # Max turns for the interaction
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config=False,
        function_map={
            search_research_papers_tool_schema["name"]: search_research_papers
        }
    )

    # Clear previous chat messages if agents are reused (not strictly necessary if re-initialized)
    # user_proxy.reset()
    # assistant.reset()

    print(f"💬 UserQueryProxy initiating chat with PaperSearchAssistant for task: '{task_message}'")
    user_proxy.initiate_chat(
        recipient=assistant,
        message=task_message,
        # clear_history=True # Ensures clean slate for this specific chat
    )
    print(f"🏁 Chat completed for task: '{task_message}'")

    # Extract the final response from the assistant
    # The last message *sent by the assistant* is what we consider its final response.
    # However, initiate_chat stores messages *received by the user_proxy*.
    # The last message in user_proxy.chat_messages[assistant] is the last one received by user_proxy *from* assistant.
    
    agent_final_response = "No response from assistant." # Default
    conversation_history = []

    if assistant in user_proxy.chat_messages and user_proxy.chat_messages[assistant]:
        conversation_history = user_proxy.chat_messages[assistant] # Get all messages from assistant to proxy
        
        # Find the last message from the assistant to the user proxy
        # This might be tricky if the chat ends differently. A robust way is to look at the full convo.
        # For simplicity, let's assume the last message TO the user_proxy from assistant is the one.
        # Or even better, get all messages and filter.

        # Let's get the full conversation history from the user_proxy's perspective
        # This will include messages from UserQueryProxy to PaperSearchAssistant and vice-versa.
        # We need to ensure the history includes messages in chronological order.
        # Autogen's `chat_messages` stores messages *received by* the agent.
        # `user_proxy.chat_messages[assistant]` = messages assistant sent to user_proxy
        # `assistant.chat_messages[user_proxy]` = messages user_proxy sent to assistant
        
        # A more complete history:
        full_chronological_history = []
        # This requires a bit more work to interleave properly or just use one side's view
        # For the critic, user_proxy.chat_messages[assistant] might be enough if it shows assistant's replies
        # But a truly full history is better. Let's build it.
        
        # Simplest approach for now: the history from the user_proxy's perspective for this one chat
        # This is a list of dicts.
        raw_history_for_critic = user_proxy.get_chat_messages(assistant)

        # Extract the agent's *actual last utterance* to the user
        # The very last message in raw_history_for_critic should be from the assistant if it replied last.
        if raw_history_for_critic:
            # The 'content' of the last message sent by the assistant
            # But we need to be careful: the last message could be a tool call response, not a user-facing message.
            # We want the last *user-facing* message from the assistant.
            
            # Iterate backwards to find the last message from assistant that isn't a tool call itself
            for msg in reversed(raw_history_for_critic):
                if msg.get("role") == "assistant" and not msg.get("tool_calls"): # Adjust if tool_calls structure is different
                    if isinstance(msg.get("content"), str):
                        agent_final_response = msg["content"]
                        break
            # If the loop finishes without finding one, agent_final_response remains default
            
    # For the critic, we'll pass the raw_history_for_critic which is user_proxy.get_chat_messages(assistant)
    # This function returns a list of messages exchanged with that specific agent.
    return agent_final_response, raw_history_for_critic

if __name__ == "__main__":
    task = "Hi, can you find me 2 research papers on 'quantum machine learning' published after 2022 with at least 10 citations?"
    print(f"CHATSTART!!! Starting chat for task: {task}\n")
    user_proxy.initiate_chat(
        recipient=assistant,
        message=task
    )
    print("\nChat finished.")