import json
from main_agent import run_paper_search_chat
from evaluation import evaluate_agent_response

# --- Test Prompts (from Phase 4, Step 2) ---
test_prompts = [
    # A. Typical Prompts
    "Find 3 research papers on 'transformer models in NLP' published in 2021 with more than 200 citations.",
    "Show me one highly cited paper about 'CRISPR gene editing applications' published before 2019. By highly cited, I mean over 1000 citations.",
    # B. Ambiguous Prompts
    "I need some recent papers on reinforcement learning.",
    "Find good papers about AI ethics.",
    # C. Complex Requests
    "Can you get me up to 5 papers on 'graph neural networks' published after 2022, but I only want those with at least 50 citations?",
    # D. Edge Cases or Error-Inducing Prompts
    "Find research papers on 'time travel feasibility' published in the year 2500.",
    "Search for papers on '' with 10 citations.", # Empty topic
    "I want papers on 'the history of aether physics' with exactly -5 citations published before 1900."
]

all_evaluations = []

if __name__ == "__main__":
    print("--- Starting Evaluation Suite ---")

    for i, prompt in enumerate(test_prompts):
        print(f"\n\n--- Test Case {i+1}/{len(test_prompts)} ---")
        print(f"User Prompt: {prompt}")

        # 1. Run the PaperSearchAgent
        #    This might take a while for each prompt due to LLM calls & rate limits
        agent_final_response, conversation_history = run_paper_search_chat(prompt)

        print("\n--- Agent Interaction Summary ---")
        print(f"Agent's Final Response to User:\n{agent_final_response}")
        # Optionally print full history for debugging here, or rely on critic's use of it
        # print("\nFull Conversation History (for critic):")
        # for hist_msg in conversation_history:
        #     print(hist_msg)

        # 2. Run the Critic Agent
        if not conversation_history:
            print("WARNING: No conversation history was recorded. Skipping critic evaluation for this prompt.")
            evaluation_result = {"error": "No conversation history available for critic."}
        else:
            evaluation_result = evaluate_agent_response(
                user_prompt=prompt,
                agent_final_response=agent_final_response,
                conversation_history=conversation_history
            )

        print("\n--- Critic's Evaluation Result ---")
        print(json.dumps(evaluation_result, indent=2))

        all_evaluations.append({
            "prompt_id": i + 1,
            "user_prompt": prompt,
            "agent_final_response": agent_final_response,
            # "conversation_history": conversation_history, # Potentially very long, maybe omit from summary
            "critic_evaluation": evaluation_result
        })
        
        # Optional: Save results progressively to a file
        with open("evaluation_results.jsonl", "a") as f: # Append mode, JSON Lines format
            f.write(json.dumps(all_evaluations[-1]) + "\n")

    print("\n\n--- Evaluation Suite Finished ---")
    # print("\n--- All Evaluation Results ---")
    # print(json.dumps(all_evaluations, indent=2)) # This can be very long

    # You can now analyze the "evaluation_results.jsonl" file.
    print(f"\nAll evaluation results saved to evaluation_results.jsonl")