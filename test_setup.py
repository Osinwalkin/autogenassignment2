from autogen.agentchat import AssistantAgent, UserProxyAgent
from config import LLM_CONFIG # Import the config from your config.py

print("Attempting to create agents...")

try:
    # Create a simple Assistant Agent
    assistant = AssistantAgent(
        name="Assistant",
        llm_config=LLM_CONFIG
    )

    # Create a User Proxy Agent
    user_proxy = UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER", # Don't ask for input
        max_consecutive_auto_reply=1, # Prevent loops
        code_execution_config=False, # Disable code execution for this simple test
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"), # Define how it detects termination
        system_message="You are a user proxy. Reply TERMINATE when the task is done.",
    )

    print("Agents created successfully.")

    # Initiate a very simple chat
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