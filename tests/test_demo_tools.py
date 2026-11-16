"""Tool registration and real argument plumbing; Mac app calls are mocked."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from test_failures import ROOT, load_upstream
from demo_config import CONTEXT_LENGTH, ENABLED_APPS, ENABLED_TOOLS, configure_tools, select_tool_names
models, TinyAgent, _ = load_upstream()
from src.llm_compiler.task_fetching_unit import TaskFetchingUnit
from src.tiny_agent.tools.contacts import Contacts
from src.tiny_agent.models import TransportationOptions


def configured_agent():
    model = models.ModelConfig(api_key="local-unused", context_length=CONTEXT_LENGTH,
        model_name="tinyagent-7b", model_type=models.ModelType.LOCAL, tokenizer=object(), port=8080)
    config = models.TinyAgentConfig(apps={models.App(name) for name in ENABLED_APPS},
        custom_instructions="", llmcompiler_config=model, sub_agent_config=model,
        embedding_model_config=None, azure_api_version=None, azure_endpoint=None,
        hf_token=None, zoom_access_token=None,
        whisper_config=models.WhisperConfig(models.ModelType.LOCAL, None, 8081))
    agent = TinyAgent(config)
    configure_tools(agent, config, "", False)
    return agent


async def execute(agent, plan):
    tasks = agent.agent.planner.output_parser.parse(plan)
    scheduler = TaskFetchingUnit()
    scheduler.set_tasks(tasks)
    await asyncio.wait_for(scheduler.schedule(), timeout=2)
    return tasks


class DemoToolTests(unittest.TestCase):
    def test_notes_request_does_not_offer_pdf_tools(self):
        chosen = select_tool_names("Open Demo brief from Notes, create a checklist note, and remind me tomorrow.")
        self.assertIn("open_note", chosen)
        self.assertIn("create_reminder", chosen)
        self.assertNotIn("summarize_pdf", chosen)
        self.assertNotIn("open_and_get_file_path", chosen)

    def test_outing_request_offers_the_four_required_tools(self):
        self.assertEqual(select_tool_names("Find Alex's email in Contacts, get walking directions, draft an email and attach Museum itinerary.pdf."),
            {"get_email_address", "maps_show_directions", "compose_new_email", "open_and_get_file_path"})

    def test_demo_toolset_exposes_required_functions_only(self):
        agent = configured_agent()
        tools = {tool.name for tool in agent.agent.planner.tools}
        self.assertEqual(tools, ENABLED_TOOLS)
        self.assertNotIn("send_sms", tools)
        self.assertNotIn("reply_to_email", tools)
        self.assertNotIn("get_walking_directions", tools)
        for name in tools:
            self.assertIn(name + "(", agent.agent.planner.system_prompt)

    def test_directions_contact_and_attachment_reach_mail_draft(self):
        agent = configured_agent()
        agent.compose_email_agent.query = "Draft an email containing the walking-directions link."
        path = str(ROOT / "demo-files/Museum itinerary.pdf")
        route = "https://maps.apple.com/?saddr=Berlin+Hauptbahnhof&daddr=Museum+Island&dirflg=w"
        async def compose(context, email_thread="", mode=None):
            self.assertEqual(context, "Directions: " + route)
            return "Here is our walking route: " + route
        plan = '''1. get_email_address("Alex Demo")
2. maps_show_directions("Berlin Hauptbahnhof", "Museum Island", "w")
3. open_and_get_file_path("Museum itinerary.pdf")
4. compose_new_email(["$1"], [], "Museum walk", "$2", ["$3"])
5. join()
'''
        with patch.object(agent.computer.contacts, "get_email_address", return_value="alex@example.invalid"), \
             patch.object(agent.computer.maps, "show_directions", return_value="Directions: " + route) as directions, \
             patch.object(agent.computer.spotlight_search, "open", return_value=path), \
             patch("src.tiny_agent.sub_agents.compose_email_agent.ComposeEmailAgent.__call__", side_effect=compose), \
             patch("src.tiny_agent.tools.mail.run_applescript", return_value="") as script:
            tasks = asyncio.run(execute(agent, plan))
        directions.assert_called_once_with(end="Museum Island", start="Berlin Hauptbahnhof", transport=TransportationOptions.WALKING)
        written = script.call_args.args[0]
        self.assertIn("alex@example.invalid", written)
        self.assertIn(path, written)
        self.assertIn(route, written)
        self.assertNotIn("send newMessage", written)
        self.assertIn("successfully", tasks[4].observation)

    def test_pdf_summary_is_passed_to_note_and_reminder(self):
        agent = configured_agent()
        summary = "Studio B: bring the laptop, adapter, and five handouts."
        path = str(ROOT / "demo-files/Workshop brief.pdf")
        plan = '''1. open_and_get_file_path("Workshop brief.pdf")
2. summarize_pdf("$1")
3. create_note("Workshop summary", "$2", "TinyAgent Demo")
4. create_reminder("Review Workshop summary", "2026-11-17 09:00:00", "$2", "", 0, False)
5. join()
'''
        with patch.object(agent.computer.spotlight_search, "open", return_value=path), \
             patch("src.tiny_agent.sub_agents.pdf_summarizer_agent.PDFSummarizerAgent.__call__", new=AsyncMock(return_value=summary)) as summarize, \
             patch("src.tiny_agent.sub_agents.notes_agent.NotesAgent.__call__", new=AsyncMock(return_value="<h1>Summary</h1><p>"+summary+"</p>")) as format_note, \
             patch.object(agent.computer.notes, "create_note", return_value="Note created successfully"), \
             patch.object(agent.computer.reminders, "create_reminder", return_value="Reminder created successfully") as reminder:
            tasks = asyncio.run(execute(agent, plan))
        summarize.assert_awaited_once_with(path)
        self.assertEqual(format_note.call_args.args[1], summary)
        self.assertEqual(reminder.call_args.kwargs["notes"], summary)
        self.assertEqual(tasks[4].args[2], summary)

if __name__ == "__main__":
    unittest.main()
