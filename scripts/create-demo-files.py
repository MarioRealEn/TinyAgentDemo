"""Create the two short sample PDFs without replacing existing files."""
from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parents[1] / "demo-files"
SAMPLES = {
    "Workshop brief.pdf": (
        "Workshop brief",
        "Fictional workshop for the TinyAgent demo.\n\n"
        "Topic: Running AI assistants locally on a Mac.\n"
        "Location: Studio B. Session: 10 AM to noon.\n\n"
        "Before the workshop, prepare the slide deck, charge the laptop, test the USB-C "
        "projector adapter, print five one-page handouts, and rehearse a three-minute introduction.\n\n"
        "The workshop will demonstrate creating a note, scheduling a reminder, and "
        "preparing an email draft. The audience should see actual tool results."
    ),
    "Museum itinerary.pdf": (
        "Museum itinerary",
        "Fictional outing for the TinyAgent demo.\n\n"
        "Meeting point: Berlin Hauptbahnhof.\n"
        "Destination: Museum Island, Berlin.\n"
        "Transport: walking.\n\n"
        "Plan a relaxed walk, take comfortable shoes, and bring water. "
        "Confirm the meeting time and museum opening hours separately. "
        "The email draft should include an Apple Maps walking-directions link "
        "and this itinerary as an attachment."
    ),
}


def main():
    ROOT.mkdir(exist_ok=True)
    for filename, (title, body) in SAMPLES.items():
        path = ROOT / filename
        if path.exists():
            continue
        with fitz.open() as document:
            page = document.new_page()
            page.insert_text((50, 65), title, fontsize=20)
            remaining = page.insert_textbox(fitz.Rect(50, 95, 545, 760), body, fontsize=12)
            if remaining < 0:
                raise RuntimeError(f"The sample text did not fit: {filename}")
            document.save(path)
        print(f"Created {path}")


if __name__ == "__main__":
    main()
