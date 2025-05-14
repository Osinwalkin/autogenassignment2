import json
from main_agent import create_paper_search_agents, run_paper_search_chat
from evaluation import evaluate_agent_response

# --- Test Prompts (from Phase 4, Step 2) ---
TEST_PROMPTS_FULL = [
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

# Configuration for prompts
# 2 = "I need some recent papers on reinforcement learning."
test_indices_to_run = [2]

OUTPUT_FILE = "evaluation_results.jsonl"

all_evaluations = []

def main():
    print("--- Starting Evaluation Suite ---")

    # Create agents once to be used for all test cases
    # The run_paper_search_chat function will reset them for each chat.
    print("🔧 Initializing agents for the evaluation suite...")
    user_proxy, assistant = create_paper_search_agents()
    print("✅ Agents initialized.")

    if test_indices_to_run is not None:
        prompts_to_run = [TEST_PROMPTS_FULL[i] for i in test_indices_to_run]
        print(f"Running a subset of {len(prompts_to_run)} test prompts.")
    else:
        prompts_to_run = TEST_PROMPTS_FULL
        print(f"Running all {len(prompts_to_run)} test prompts.")

    all_evaluations_summary = [] # To store just the critic's output for final summary

    # Clear the output file at the start of a new suite run
    with open(OUTPUT_FILE, "w") as f:
        pass # Creates or truncates the file

    for i, prompt_text in enumerate(prompts_to_run):
        current_prompt_index = test_indices_to_run[i] if test_indices_to_run else i
        print(f"\n\n--- Test Case {i+1}/{len(prompts_to_run)} (Overall Index: {current_prompt_index + 1}) ---")
        print(f"User Prompt: {prompt_text}")

        try:
            # 1. Run the PaperSearchAgent
            agent_final_response, conversation_history = run_paper_search_chat(
                task_message=prompt_text,
                user_proxy=user_proxy, # Pass the created agents
                assistant=assistant   # Pass the created agents
            )

            print("\n--- Agent Interaction Summary (for this test case) ---")
            print(f"Agent's Final User-Facing Response:\n{agent_final_response}")

            # 2. Run the Critic Agent
            if not conversation_history:
                print("WARNING: No conversation history was recorded. Skipping critic evaluation for this prompt.")
                evaluation_result = {"error": "No conversation history available for critic."}
            else:
                evaluation_result = evaluate_agent_response(
                    user_prompt=prompt_text,
                    agent_final_response=agent_final_response,
                    conversation_history=conversation_history
                )

            print("\n--- Critic's Evaluation Result (for this test case) ---")
            print(json.dumps(evaluation_result, indent=2))

            current_evaluation_data = {
                "prompt_id_in_run": i + 1,
                "overall_prompt_id": current_prompt_index + 1,
                "user_prompt": prompt_text,
                "agent_final_response": agent_final_response,
                "critic_evaluation": evaluation_result
                # "conversation_history": conversation_history, # Exclude for brevity in file, but useful for deep dive
            }
            all_evaluations_summary.append(evaluation_result) # Store just critic output for final summary

        except Exception as e:
            print(f"ERROR occurred during Test Case {i+1} for prompt: '{prompt_text}'")
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc()
            current_evaluation_data = {
                "prompt_id_in_run": i + 1,
                "overall_prompt_id": current_prompt_index + 1,
                "user_prompt": prompt_text,
                "error_during_processing": str(e),
                "traceback": traceback.format_exc()
            }
            # Still try to save error information
        
        # Save results progressively to a file
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(current_evaluation_data) + "\n")
        print(f"Results for Test Case {i+1} appended to {OUTPUT_FILE}")


    print("\n\n--- Evaluation Suite Finished ---")
    
    # Basic summary of scores (if evaluations were successful)
    print("\n--- Overall Score Summary (from successful evaluations) ---")
    successful_evals = [e for e in all_evaluations_summary if e and "error" not in e]
    if successful_evals:
        criteria_keys = [
            "completeness_score", "quality_accuracy_score", "robustness_score",
            "tool_usage_score", "efficiency_conciseness_score"
        ]
        for key in criteria_keys:
            scores = [e.get(key, 0) for e in successful_evals if isinstance(e.get(key), int)]
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"Average {key}: {avg_score:.2f}")
            else:
                print(f"No scores found for {key}")
    else:
        print("No successful evaluations to summarize.")

    print(f"\nAll evaluation details saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()