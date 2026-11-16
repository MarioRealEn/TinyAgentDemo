"""CLI adapter for upstream TinyAgent; uses local inference and its original tools."""
import argparse
import asyncio
import datetime
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
import warnings
from functools import wraps
from demo_config import CONTEXT_LENGTH, ENABLED_APPS, configure_tools

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".tools" / "cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(ROOT / "vendor" / "TinyAgent"))


def load_upstream():
    if not (ROOT / "vendor" / "TinyAgent" / "src").is_dir():
        raise RuntimeError("Upstream code missing. Run ./scripts/setup.sh first.")
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
    warnings.filterwarnings("ignore", message="Importing verbose from langchain root module.*")
    # Redirect upstream runtime artifacts before importing modules that use this constant.
    from src.tiny_agent import models
    models.TINY_AGENT_DIR = str(ROOT / "runtime")
    from src.tiny_agent.tiny_agent import TinyAgent
    from src.utils.logger_utils import enable_logging
    enable_logging(False)
    from transformers import AutoTokenizer
    return models, TinyAgent, AutoTokenizer


async def run(args):
    models, TinyAgent, AutoTokenizer = load_upstream()
    tokenizer_path = ROOT / "models" / f"tokenizer-{args.model.removesuffix('-f16')}"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, use_fast=False)
    if args.check:
        # Import and tokenizer checks only; this does not execute tools or run inference.
        for size in ("7b", "1.1b"):
            AutoTokenizer.from_pretrained(str(ROOT / "models" / f"tokenizer-{size}"),
                                          local_files_only=True, use_fast=False)
        print("Upstream TinyAgent imports and both tokenizers loaded successfully.")
        return
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=5) as response:
            available = {item["id"] for item in json.load(response)["data"]}
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Start ./scripts/serve.sh {args.model} in another terminal.") from error
    name = f"tinyagent-{args.model}"
    if name not in available:
        raise RuntimeError(f"Expected {name}, server provides {sorted(available)}. Restart it with the matching model.")
    model_config = models.ModelConfig(api_key="local-unused", context_length=CONTEXT_LENGTH,
        model_name=name, model_type=models.ModelType.LOCAL, tokenizer=tokenizer, port=8080)
    # Construct the upstream dataclass directly, bypassing the desktop config loader's
    # unconditional API-key requirement. Text input does not call Whisper.
    config = models.TinyAgentConfig(
        apps={models.App(name) for name in ENABLED_APPS},
        custom_instructions=f"The local date and time is {datetime.datetime.now().astimezone().isoformat()}.",
        llmcompiler_config=model_config, sub_agent_config=model_config,
        embedding_model_config=None, azure_api_version=None, azure_endpoint=None,
        hf_token=None, zoom_access_token=None,
        whisper_config=models.WhisperConfig(provider=models.ModelType.LOCAL, api_key=None, port=8081))
    agent = TinyAgent(config)
    configure_tools(agent, config, args.query, args.retrieve_examples)
    # Validate the complete plan before any real application action starts.
    agent.agent.planner_stream = False
    if plan_path := os.environ.get("TINYAGENT_PLAN_PATH"):
        original_run_llm = agent.agent.planner.run_llm
        async def record_plan(*values, **kwargs):
            text = await original_run_llm(*values, **kwargs)
            with Path(plan_path).open("a") as output:
                output.write(text + "\n")
            return text
        agent.agent.planner.run_llm = record_plan
    if args.execute:
        events = []
        def traced_tool(tool):
            original = tool.func
            @wraps(original)
            async def invoke(*values, **kwargs):
                event = {"tool": tool.name, "args": values}
                events.append(event)
                try:
                    result = await original(*values, **kwargs)
                    event["result"] = result
                    event["failed"] = str(result).startswith(("Failed", "Error", "Command '['"))
                    print(f"\nTool result ({tool.name}): {result}", flush=True)
                    return result
                except Exception as error:
                    event.update(result=str(error), failed=True)
                    raise
            return invoke
        for tool in agent.agent.planner.tools:
            tool.func = traced_tool(tool)
        try:
            response = await asyncio.wait_for(agent.arun(args.query), timeout=180)
            print(f"\nAgent response: {response}")
        finally:
            trace = json.dumps(events, indent=2) + "\n"
            (ROOT / "runtime" / "last-execution.json").write_text(trace)
            if trace_path := os.environ.get("TINYAGENT_TRACE_PATH"):
                Path(trace_path).write_text(trace)
        if not events or any(event.get("failed", True) for event in events):
            raise RuntimeError("A tool failed or no tool was executed; see runtime/last-execution.json.")
    else:
        print("Generating a plan with the upstream planner (no tools executed):", flush=True)
        plan = await asyncio.wait_for(agent.agent.planner.run_llm(
            inputs={"input": args.query}, is_replan=False), timeout=180)
        if not plan or not plan.strip():
            raise RuntimeError("The model returned an empty plan.")
        agent.agent.planner.output_parser.parse(plan + "\n")
        print(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["7b", "1.1b", "1.1b-f16"], default="7b")
    parser.add_argument("--execute", action="store_true", help="Execute the generated plan using the enabled demo tools")
    parser.add_argument("--check", action="store_true", help="Check imports and tokenizers without inference")
    parser.add_argument("--retrieve-examples", action="store_true", help="Select relevant bundled examples locally by word overlap")
    parser.add_argument("query", nargs="?", default="Create a reminder titled TinyAgent demo.")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except (RuntimeError, ImportError, OSError, ValueError) as error:
        print(f"Demo error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
