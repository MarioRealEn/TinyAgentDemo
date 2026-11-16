# TinyAgent: local Mac demo

A small browser interface for running the published **TinyAgent-1.1B** and
**TinyAgent-7B** models locally on an Apple Silicon Mac. Enter a natural-language
request, inspect a proposed plan, or execute tools that interact with real Mac
apps.

This repository is an **adapted demonstration of the authors' code and models**.
It is not an official TinyAgent release or a reproduction of the paper's reported
accuracy. In particular, **the original DeBERTa ToolRAG is not enabled**: this
wrapper uses keyword filtering and example selection by word overlap.

## Paper and citation

Lutfi Eren Erdogan, Nicholas Lee, Siddharth Jha, Sehoon Kim, Ryan Tabrizi, Suhong
Moon, Coleman Richard Charles Hooper, Gopala Anumanchipalli, Kurt Keutzer, and Amir
Gholami. 2024. **TinyAgent: Function Calling at the Edge.** In *Proceedings of the
2024 Conference on Empirical Methods in Natural Language Processing: System
Demonstrations*, pages 80–88. Association for Computational Linguistics.

- [Published paper and citation](https://aclanthology.org/2024.emnlp-demo.9/)
- [DOI: 10.18653/v1/2024.emnlp-demo.9](https://doi.org/10.18653/v1/2024.emnlp-demo.9)
- [arXiv version](https://arxiv.org/abs/2409.00608)
- [Authors' source code](https://github.com/SqueezeAILab/TinyAgent)
- [Berkeley research blog](https://bair.berkeley.edu/blog/2024/05/29/tiny-agent/)

```bibtex
@inproceedings{erdogan-etal-2024-tinyagent,
  title = "{T}iny{A}gent: Function Calling at the Edge",
  author = "Erdogan, Lutfi Eren and
    Lee, Nicholas and
    Jha, Siddharth and
    Kim, Sehoon and
    Tabrizi, Ryan and
    Moon, Suhong and
    Hooper, Coleman Richard Charles and
    Anumanchipalli, Gopala and
    Keutzer, Kurt and
    Gholami, Amir",
  booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
  year = "2024",
  month = nov,
  address = "Miami, Florida, USA",
  publisher = "Association for Computational Linguistics",
  pages = "80--88",
  doi = "10.18653/v1/2024.emnlp-demo.9",
  url = "https://aclanthology.org/2024.emnlp-demo.9/"
}
```

The paper investigates whether small, task-specific language models can plan
function calls on a local device. The authors fine-tune models to select tools,
fill their arguments, and express dependencies using LLMCompiler. Their system
also uses learned tool retrieval and quantization to reduce inference costs.
The published performance claims concern the authors' evaluation; they should
not be attributed to this wrapper.

## What this demo does

The selected TinyAgent model generates a plan using predefined functions.
Python parses the plan and calls the functions; the model does not directly
control the operating system or generate arbitrary AppleScript.

Results can flow between calls. A reference such as `$2` means “use the result
of action 2.” Before a dependent call executes, the executor replaces that
reference with the actual result. This allows a PDF summary to become a note or
reminder body, or a contact lookup to supply an email recipient.

The interface offers:

- **Preview plan:** generate and validate a plan without executing its tools.
- **Run with tools:** execute the plan and display returned tool results.
- Three selectable local models, loaded automatically when needed.
- Example prompts for the workflows below.

Each submitted prompt is a new task. The displayed history is not conversation
memory: a later prompt must read a saved note again if it needs that content.
Refreshing the page clears the displayed history; local run logs remain.

## How tools and examples are selected

The current path is implemented in [demo_config.py](demo_config.py):

1. A hand-written keyword filter chooses which tool descriptions to expose.
   For example, “walking” enables Maps directions and “remind” enables Reminders.
   If no category matches, all nine enabled tools are exposed.
2. The wrapper reads the authors' bundled examples and keeps examples whose
   tools are in the selected set. In the browser app, it ranks those examples by
   shared words with the request and includes up to six in the prompt.
3. The selected TinyAgent model generates the actual calls, arguments, and
   dependency references. The wrapper does not directly execute stored example
   plans as answers.
4. The complete plan is validated before dispatch, and the original tool
   implementations perform the actions, with the compatibility fixes listed below.

The upstream repository includes roughly 10 MB of precomputed embeddings in
`src/tiny_agent/tool_rag/text-embedding-3-small/embeddings.pkl`. The inspected
checkout contains 1,512 entries with example text, vectors, and tool labels.
**This demo reads the text and labels but ignores the vectors.** Neither DeBERTa
classification nor embedding-similarity retrieval is used here. The CLI's
`--retrieve-examples` flag means word-overlap ranking, not the authors' ToolRAG.

**Optional upgrade: the authors' trained DeBERTa classifier is available online**
at [TinyAgent-ToolRAG on Hugging Face](https://huggingface.co/squeeze-ai-lab/TinyAgent-ToolRAG)
(approximately 579 MB including tokenizer files). Integrating it with the original
retrieval pipeline could improve tool-selection accuracy and downstream plan
success compared with this demo's keyword heuristics. The authors report improved
results with their learned ToolRAG in the [research blog](https://bair.berkeley.edu/blog/2024/05/29/tiny-agent/).
We have not measured that improvement in this wrapper. This optional upgrade
requires configuring the classifier and compatible query-embedding retrieval;
it is not downloaded or enabled by the current setup script.

These are few-shot, assisted demonstrations. A model can copy or adapt the
provided examples, and the custom filter can exclude a needed tool when phrasing
does not match its rules. Successful prepared demos do not establish performance
on unseen tasks. No comparison against Siri has been measured.

## Requirements

- An Apple Silicon Mac running macOS. The setup script targets `arm64` macOS.
- Xcode Command Line Tools, including the compiler and macOS SDK. Install them
  with `xcode-select --install` if needed.
- Git and an internet connection for initial setup.
- Approximately **7.2 GB for model weights**, plus space for dependencies,
  caches, source checkouts, and the llama.cpp build.
- The relevant Mac apps and accounts configured for the actions you want to run.

The demo was tested on an M3 Max MacBook Pro with 36 GB of unified memory.
Other configurations have not been systematically tested. No OpenAI API key,
Conda installation, model training, or separate model-serving application is
required.

## Setup

Clone or copy this repository, then run from its root:

```sh
chmod +x scripts/*.sh "Start TinyAgent.command"
./scripts/setup.sh
```

Setup installs Python 3.11 and dependencies inside the project, clones TinyAgent
and llama.cpp into `vendor/`, applies the local compatibility patches, downloads
and verifies the models, creates two sample PDFs, and builds llama-server with
Metal support. It does not modify your shell profile or require sudo.

| Model | Weight format | Approximate download | Model selector |
| --- | --- | --- | --- |
| TinyAgent-7B | Q4_K_M, 4-bit | 4.37 GB | `7b` |
| TinyAgent-1.1B | Q4_K_M, 4-bit | 668 MB | `1.1b` |
| TinyAgent-1.1B | F16, 16-bit | 2.2 GB | `1.1b-f16` |

Official weights: [7B GGUF](https://huggingface.co/squeeze-ai-lab/TinyAgent-7B-GGUF)
and [1.1B GGUF](https://huggingface.co/squeeze-ai-lab/TinyAgent-1.1B-GGUF).
The downloader also retrieves the matching base-model tokenizers. It checks the
weight files against their published SHA-256 hashes and retains completed chunks
for resuming interrupted downloads. To resume setup, run the same command again.

Setup currently fetches upstream revisions at installation time rather than
using a complete reproducible lockfile. Resolved source commits, model revisions,
hashes, and installed dependency versions are recorded in `runtime/`. The SDK
selection includes a workaround for the development machine: it selects the
installed macOS 26.5 SDK when available and otherwise uses `xcrun`'s default.

## Open the app

Double-click **Start TinyAgent.command**, or run:

```sh
./"Start TinyAgent.command"
```

The launcher opens **http://127.0.0.1:8765** in your browser. Choose a model, type
a request, and select **Preview plan** or **Run with tools**. Keep the launcher's
Terminal window open while using it. Ctrl-C stops the app and its owned model
server. Opening the launcher again reuses an existing app instance.

The interface binds to loopback only; it is not a hosted or multi-user service.
Model requests go to a local llama.cpp server on port 8080. The wrapper uses an
8192-token context and runs one task/model at a time. Stop a manually launched
server on port 8080 before asking the interface to switch models.

macOS may ask permission for Terminal to control Notes, Reminders, Contacts, Mail,
or System Events. Grant access for the apps you intend to demonstrate. Mac app
integration is separate from model inference: Maps and iCloud synchronization can
use the network even though model inference is local. Setup does not disable
network connectivity.

## Enabled tools

| Area | Original functions exposed by the wrapper |
| --- | --- |
| Notes | `open_note`, `create_note`, `append_note_content` |
| Reminders | `create_reminder` |
| Files and PDFs | `open_and_get_file_path`, `summarize_pdf` |
| Contacts | `get_email_address` |
| Maps | `maps_show_directions` |
| Mail drafts | `compose_new_email` |

The draft tool composes an outgoing message without sending it. Calendar, Zoom,
SMS, and Mail reply/forward are not enabled. Voice input and the authors' packaged
Mac app are not configured.

## Demo prompts

Start with a preview and inspect the resulting calls. Running a prompt can create
real notes, reminders, or email drafts. Repeating it can create duplicates.

### 1. Notes brief to checklist and reminder

Create the sample brief:

```sh
osascript scripts/seed-cross-app-demo.applescript
```

The Notes tools currently target the account named **iCloud**. Ensure the
`TinyAgent Demo` folder and `Demo brief` note are in that account; the seed script
creates them in the default Notes account. A different default account may require
moving the sample note/folder to iCloud.

> Open Demo brief from the TinyAgent Demo folder. Use the returned text to create
> a concise preparation checklist in a new note named Preparation checklist in
> that folder. Then open the new checklist note and use its contents as the notes
> of a reminder titled Review preparation checklist, due tomorrow at 9 AM.

Expected flow: read the brief, create the checklist, read the checklist, and pass
its text into a reminder. Check the actual outputs for the source brief's five
requirements; a generic checklist is not a successful completion.

### 2. Local PDF to summary and follow-up

Setup creates `demo-files/Workshop brief.pdf`. The interface's PDF example button
inserts its full path, so it does not depend on Spotlight having indexed it.

> Find Workshop brief.pdf, summarize it into a note called Workshop summary in
> the TinyAgent Demo folder, and remind me tomorrow at 9 AM to review it. Include
> the summary in the reminder.

If using the prompt manually, substitute the PDF's full path when filename lookup
fails. Expected flow: resolve the PDF path, summarize the document, and pass the
returned summary to both note creation and reminder creation. Inspect the reminder
body: a generic “review the summary” sentence does not satisfy this request.

### 3. Contact and walking directions to email draft

Setup creates `demo-files/Museum itinerary.pdf`. **Alex Demo is a placeholder**:
replace it with an existing contact that has an email address, or create that demo
contact yourself. The wrapper does not create contacts automatically.

> Find Alex Demo's email address in Contacts. Get walking directions from Berlin
> Hauptbahnhof to Museum Island. Draft an email to Alex with the directions link
> and attach Museum itinerary.pdf.

Expected flow: look up the recipient, obtain an Apple Maps walking-directions
link, locate the PDF, and compose a draft containing the link and attachment.
The actual Maps tool is `maps_show_directions` with transport `"w"`;
`get_walking_directions` is not a registered tool. The interface's example button
inserts the attachment's full path. Check recipient, body, and attachment in Mail.

## Command-line use

In one terminal:

```sh
./scripts/serve.sh 7b
```

In a second terminal, preview a plan:

```sh
./scripts/demo.sh --model 7b --retrieve-examples \
  'Create a reminder titled Rehearse the presentation for tomorrow at 9 AM.'
```

Add `--execute` to perform the actions:

```sh
./scripts/demo.sh --model 7b --retrieve-examples --execute \
  'Open the note Demo brief in the TinyAgent Demo folder.'
```

Use `1.1b` or `1.1b-f16` in both commands to select another model. CLI requests do
not inherit previous requests' results.

## Local changes to the original project

This repository adds a browser interface, launcher, model lifecycle management,
download/setup scripts, sample documents, and the custom tool/example selection
in [demo_config.py](demo_config.py). It does not fine-tune or modify the published
model weights. The same selected model is used for planning and the existing
note-writing, PDF-summary, and email-writing sub-agents.

Setup applies four patches to the cloned authors' source:

| Patch | Purpose |
| --- | --- |
| [macos-reminders.patch](patches/macos-reminders.patch) | Construct reminder dates numerically to avoid locale parsing failures; use the parsed date argument. |
| [note-task-order.patch](patches/note-task-order.patch) | Order reads/writes to the same named note when no explicit result reference establishes that dependency. |
| [notes-lookup.patch](patches/notes-lookup.patch) | Search exact titles across iCloud when no folder is supplied; report missing/ambiguous matches and AppleScript errors. |
| [plan-validation.patch](patches/plan-validation.patch) | Reject invalid action numbering, missing/future references, wrong argument counts, and missing/final-position joins. |

The wrapper disables dispatch of partially streamed plans so the complete plan
can be validated first. It constructs configuration directly to avoid the
upstream desktop loader's API-key requirement in local mode. Contacts, Maps, Mail,
and Spotlight tool implementations remain as provided by the upstream checkout.

## Verification and limitations

Run the regression tests after setup:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

At the last check, **15 tests passed**. They cover malformed-plan handling, Notes
lookup, tool availability, and result/argument passing for the new workflows.
The new Contacts/Maps/Mail workflow tests mock Mac app side effects; they are not
evidence that every live account/permission combination has been tested.

Live checks on the development Mac confirmed model loading and simple plan
generation with all three weight variants. A local 7B run also created a checklist
from a real Notes brief and a real reminder containing that checklist, verified by
independent app readback. A later preview generated valid Contacts → Maps → file
→ email-draft calls. These are individual demonstrations, not a held-out benchmark
or a claim that every workflow succeeds on every model.

Known limitations include invented tool names, incorrect arguments, generic or
hallucinated content, and failure to include the required result references.
A valid plan and successful API calls do not prove the requested content is
correct. The agent's final “Task completed” message is not independent verification.
A failed task may already have completed earlier actions; inspect the apps before
retrying. The current Notes integration assumes an iCloud account, and the
original Contacts/Mail/file tools retain their upstream behavior and limitations.

For research comparisons, disclose the example retrieval and custom rules, use
unseen requests, and evaluate actual outputs as well as plan validity. To reproduce
the authors' retrieval pipeline, configure their DeBERTa classifier and matching
query-embedding retrieval; those components are not enabled by this wrapper.

## Logs and sharing

Local UI runs are saved under `runtime/ui-runs/`:

| File | Contents |
| --- | --- |
| `<run-id>.json` | Prompt, selected model/tools, mode, timestamps, and status. |
| `<run-id>.plan.log` | Raw generated plans. |
| `<run-id>.execution.json` | Executed calls, resolved arguments, and returned results for execution runs. |
| `<run-id>.log` / `<run-id>.stderr.log` | Process output and errors. |

Logs may contain contact addresses, note contents, document text, and local file
paths. They are local development artifacts and are ignored by Git, along with
`models/`, `vendor/`, `.venv/`, and `.tools/`. Share the wrapper source, scripts,
patches, tests, and sample documents; colleagues can recreate the environment with
setup. A manually created ZIP does not honor `.gitignore`, so exclude those local
directories when copying the project for sharing.

Upstream TinyAgent is distributed under the [authors' MIT license](https://github.com/SqueezeAILab/TinyAgent/blob/main/LICENSE).
Its copyright/license notices remain in the downloaded checkout. See the
respective model and dependency repositories for their own licensing information.
