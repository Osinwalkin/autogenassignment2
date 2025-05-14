# Autogen AI Agent til søgning af research papirer + evaluering

Både cyrstor og Osinwalkin er Christian BF. (Jeg kom til at pushe fra en forkert account)

Dette projekt implementerer en AI agent ved brug af Autogen frameworket til at søge efter research papirer baseret på specificerede kriterier (topic, year, citations). Det inkluderer også et evaluerings-setup, hvor en LLM critic agent vurderer agentens performance.


## Features

*   **Research Paper Agent:**
    *   Kan søge efter research papers ved brug af Semantic Scholar API.
    *   Filtrerer research papirer efter topic, publication year (in/before/after), og minimum citation count.
    *   Præsenterer resultater i et json.
*   **LLM-baseret Critic Agent:**
    *   Evaluerer paper search agent's svar baseret på nogle foruddefinerede kriterier:
        *   Completeness
        *   Quality/Accuracy
        *   Robustness
        *   Tool Usage
        *   Efficiency/Conciseness
    *   Returner struktureret JSON output med scores og feedback til agenten.
*   **Automated Evaluation Suite:**
    *   Kører paper search agent op imod et foruddefineret sæt af diverse test prompts. (Flere af gangen virker ikke særlig godt med mit setup. Der sker loops regelmæssigt efter 2 test cases.)

## Projekt Struktur

* .env
* .gitignore
README.md
config.py # LLM configuration for Autogen (Mistral AI)
evaluation.py # Critic agent implementering og evaluation logic
main_agent.py # Paper search agent implementering
research_tools.py # Semantic Scholar API tool implementering og schema
requirements.txt # Python dependencies
run_evaluation_suite.py #

(test.py og test_setup.py er ikke relevante for projektet og var noget jeg kørte ved siden af for at teste mistral APIen)

## Setup

Clone the repository:

git clone https://github.com/Osinwalkin/autogenassignment2.git
cd "REPOSITORY"

Create and activate a Python virtual environment

Install dependencies

pip install -r requirements.txt
(Jeg havde nogle problemer med rækkefølgen af autogen installation så det kan være mit requirements.txt ikke virker)

Få fat i en Mistral API nøgle og brug Semantic Scholar Graph API

## Usage

### Test forskellige scripts

Test the Semantic Scholar tool:

    python research_tools.py

Test the Paper Search Agent (interactive or single query):

    python main_agent.py

Test the Critic Agent:

    python evaluation.py


### Running the Full Evaluation Suite

This will run the paper search agent against all(or specified) predefined test prompts and have the critic evaluate each one. Results are saved to `evaluation_results.jsonl`.

1.  **Configure test prompts (optional):**
    Open `run_evaluation_suite.py` and modify `test_indices_to_run` if you want to run a subset of tests. Setting it to `None` runs all.
2.  **Run the suite:**

    python run_evaluation_suite.py

Jeg vil ikke anbefale at køre mere end 3 til 4 prompts da den gratis Mistral request limit sandsynligt bliver ramt på det her setup.





OBS: Der er blevet brugt LLM til udformning af system prompts. Dette er for at hjælpe med formatering og tid.
Jeg vil sige at prompten bliver meget stor og dette kan have indflydelse på modellen der bliver brugt til evaluering og søgning (Mistral Nemo 12B), da modellen er mindre.

f.eks:

"""You are a helpful AI assistant specialized in finding research papers.
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