"""The original tools needed by the three interactive demo workflows."""
ENABLED_APPS = ("notes", "reminders", "files", "contacts", "maps", "mail")
ENABLED_TOOLS = frozenset({
    "create_note", "open_note", "append_note_content", "create_reminder",
    "open_and_get_file_path", "summarize_pdf", "get_email_address",
    "maps_show_directions", "compose_new_email",
})
CONTEXT_LENGTH = 8192


def select_tool_names(query):
    """Small local keyword filter; all nine tools remain available in the app."""
    import re
    text = query.lower()
    chosen = set()
    if re.search(r"\b(note|notes|checklist|notebook)\b", text):
        chosen.update({"open_note", "create_note", "append_note_content"})
    if re.search(r"\b(remind|reminder|reminders)\b", text):
        chosen.add("create_reminder")
    if re.search(r"\.pdf\b|\b(pdf|file|files|document|attachment|attach)\b", text):
        chosen.add("open_and_get_file_path")
        if re.search(r"\b(summarize|summarise|summary|summaries)\b", text):
            chosen.add("summarize_pdf")
    if re.search(r"\b(email|e-mail|mail|draft)\b", text):
        chosen.update({"get_email_address", "compose_new_email"})
    if re.search(r"\b(contact|contacts)\b", text):
        chosen.add("get_email_address")
    if re.search(r"\b(map|maps|route|directions|walking|walk)\b", text):
        chosen.add("maps_show_directions")
    return frozenset(chosen or ENABLED_TOOLS)


def configure_tools(agent, config, query, retrieve_examples=False):
    import pickle
    import re
    from pathlib import Path
    from src.llm_compiler.planner import generate_llm_compiler_prompt
    from src.tiny_agent.prompts import (
        PLANNER_PROMPT_REPLAN,
        get_planner_custom_instructions_prompt,
    )
    planner = agent.agent.planner
    registered = {tool.name for tool in planner.tools}
    if not ENABLED_TOOLS <= registered:
        raise RuntimeError(f"Missing demo tools: {sorted(ENABLED_TOOLS - registered)}")
    available = select_tool_names(query)
    planner.tools = [tool for tool in planner.tools if tool.name in available]
    planner.output_parser.tools = planner.tools
    examples_path = Path(__file__).parent / "vendor/TinyAgent/src/tiny_agent/tool_rag/text-embedding-3-small/embeddings.pkl"
    with examples_path.open("rb") as source:
        bundled = pickle.load(source)
    query_words = set(re.findall(r"\w+", query.lower()))
    candidates = [item for item in bundled.values() if set(item["tools"]) <= available]
    if retrieve_examples:
        candidates.sort(key=lambda item: len(query_words & set(re.findall(r"\w+", item["example"].lower()))), reverse=True)
    examples = "###\n".join(item["example"] for item in candidates[:6])
    if retrieve_examples:
        print(f"Using {min(6, len(candidates))} examples from the authors' bundled database.", flush=True)
    instructions = get_planner_custom_instructions_prompt(planner.tools, config.custom_instructions)
    instructions += (
        '\nUse only the exact tool names listed above. For walking directions, call '
        'maps_show_directions with transport="w". compose_new_email opens a draft for review. '
        'When a later tool needs earlier output, pass the corresponding $id reference; '
        'do not replace actual source content with a generic description. '
        'The third argument of create_reminder is the actual notes text. '
        'If asked to include a generated summary there, pass the summary result reference.'
    )
    planner.system_prompt = generate_llm_compiler_prompt(planner.tools, examples, instructions)
    planner.system_prompt_replan = generate_llm_compiler_prompt(
        planner.tools, PLANNER_PROMPT_REPLAN, instructions, is_replan=True)
