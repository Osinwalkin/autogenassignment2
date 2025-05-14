from autogen.agentchat import AssistantAgent, UserProxyAgent
from config import LLM_CONFIG

print("Attempting to create agents...")

try:
    assistant = AssistantAgent(
        name="Assistant",
        llm_config=LLM_CONFIG
    )

    user_proxy = UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False, 
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        system_message="You are a user proxy. Reply TERMINATE when the task is done.",
    )

    print("Agents created successfully.")

    print("Initiating test chat...")
    user_proxy.initiate_chat(
        assistant,
        message="What is the capital of France?"
    )

    print("\nTest chat finished.")
    print("Setup seems OK if no errors occurred above.")

except Exception as e:
    print(f"\nAn error occurred during setup test: {e}")
    import traceback
    traceback.print_exc()