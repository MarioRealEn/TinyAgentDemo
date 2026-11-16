"""Regressions from the failed November 15 UI runs. No Mac app writes."""
import asyncio
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo import load_upstream
load_upstream()
from langchain.schema import OutputParserException
from src.llm_compiler.output_parser import LLMCompilerPlanParser
from src.llm_compiler.task_fetching_unit import TaskFetchingUnit
from src.tiny_agent.tools.notes import Notes
from src.tools.base import Tool


class PlanValidationTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        async def create(name, content, folder):
            self.calls.append(name)
            return "created"
        async def read(name, folder):
            return "source text"
        self.parser = LLMCompilerPlanParser(tools=[
            Tool(name="create_note", func=create, description="Create note"),
            Tool(name="open_note", func=read, description="Read note")])

    def test_logged_nonconsecutive_ids_rejected_without_writes(self):
        with self.assertRaisesRegex(OutputParserException, "expected action 1, got 4"):
            self.parser.parse('4. create_note("Checklist", "text", "")\n5. join()')
        self.assertEqual(self.calls, [])

    def test_late_invalid_action_rejected_before_earlier_write(self):
        with self.assertRaises(OutputParserException):
            self.parser.parse('1. create_note("Checklist", "text", "")\n3. join()')
        self.assertEqual(self.calls, [])

    def test_future_reference_rejected(self):
        with self.assertRaisesRegex(OutputParserException, "future action"):
            self.parser.parse('1. create_note("Checklist", "$2", "")\n2. join()')

    def test_wrong_arguments_rejected(self):
        with self.assertRaisesRegex(OutputParserException, "wrong arguments"):
            self.parser.parse('1. create_note("Checklist")\n2. join()')

    def test_empty_and_unterminated_plans_rejected(self):
        for plan in ["No plan.", '1. open_note("Brief", "Demo")']:
            with self.subTest(plan=plan), self.assertRaises(OutputParserException):
                self.parser.parse(plan)

    def test_valid_dependency_plan_executes(self):
        tasks = self.parser.parse('1. open_note("Brief", "Demo")\n2. create_note("Checklist", "$1", "Demo")\n3. join()')
        self.assertEqual(tasks[2].dependencies, [1])
        async def execute():
            scheduler = TaskFetchingUnit()
            scheduler.set_tasks(tasks)
            await asyncio.wait_for(scheduler.schedule(), timeout=1)
        asyncio.run(execute())
        self.assertEqual(self.calls, ["Checklist"])
        self.assertEqual(tasks[2].args[1], "source text")


class NotesLookupTests(unittest.TestCase):
    def test_unspecified_folder_searches_account(self):
        with patch("src.tiny_agent.tools.notes.run_applescript_capture", return_value=("<p>Source brief</p>", "")) as run:
            self.assertEqual(Notes().open_note("Demo brief", "", True), "Source brief")
        script = run.call_args.args[0]
        self.assertIn('every note of account "iCloud"', script)
        self.assertNotIn('folder "Notes"', script)

    def test_explicit_folder_stays_scoped(self):
        with patch("src.tiny_agent.tools.notes.run_applescript_capture", return_value=("text", "")) as run:
            Notes().open_note("Demo brief", "TinyAgent Demo", True)
        self.assertIn('folder "TinyAgent Demo" of account "iCloud"', run.call_args.args[0])

    def test_applescript_failure_is_not_blank_success(self):
        with patch("src.tiny_agent.tools.notes.run_applescript_capture", return_value=("", "Notes permission denied")):
            self.assertEqual(Notes().open_note("Demo brief", "", True), "Error: Notes permission denied")

    def test_missing_and_ambiguous_notes_remain_errors(self):
        for message in ["Error: No note with that exact title was found in the requested location.",
                        "Error: More than one note has that title. Please specify its folder."]:
            with self.subTest(message=message), patch("src.tiny_agent.tools.notes.run_applescript_capture", return_value=(message, "")):
                self.assertEqual(Notes().open_note("Demo brief", "", True), message)


if __name__ == "__main__":
    unittest.main()
