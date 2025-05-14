import autogen
from autogen.agentchat import AssistantAgent, UserProxyAgent
from config import LLM_CONFIG
from research_tools import search_research_papers, search_research_papers_tool_schema

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
    llm_config=assistant_llm_config, # Use the new config with tools
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
    human_input_mode="TERMINATE", # Change to "ALWAYS" for interactive
    max_consecutive_auto_reply=10, # Allow for some back and forth
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,
    system_message="You are a user proxy. You will make requests to the PaperSearchAssistant. Reply TERMINATE when the task is done or no more information is needed.",
    function_map={
        search_research_papers_tool_schema["name"]: search_research_papers
    }
)

if __name__ == "__main__":
    task = "Hi, can you find me 2 research papers on 'quantum machine learning' published after 2022 with at least 10 citations?"
    print(f"CHATSTART!!! Starting chat for task: {task}\n")
    user_proxy.initiate_chat(
        recipient=assistant,
        message=task
    )
    print("\nChat finished.")