"""
Step 1: a minimal coding agent with a single tool (read_file).

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY="sk-..."

Run:
    python agent.py "Summarize the file notes.txt"
"""

import sys
import anthropic

MODEL = "claude-sonnet-5-5"
MAX_STEPS = 10  # safety limit so the loop can never run forever

client = anthropic.Anthropic()

# ---------- 1. Tool descriptions (what the model sees) ----------
TOOLS = [
    {
        "name": "read_file",
        "description": "Read the full contents of a text file and return it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"}
            },
            "required": ["path"],
        },
    }
]


# ---------- 2. Tool implementations (what actually runs) ----------
def read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


TOOL_FUNCTIONS = {"read_file": read_file}


def run_tool(name: str, args: dict) -> tuple[str, bool]:
    """Run a tool. Returns (output, is_error). Errors go back to the model."""
    if name not in TOOL_FUNCTIONS:
        return f"Unknown tool: {name}", True
    try:
        return TOOL_FUNCTIONS[name](**args), False
    except Exception as e:
        return f"{type(e).__name__}: {e}", True


# ---------- 3. Guard placeholder (the control layer will live here) ----------
class Guard:
    def check(self, user_request: str, history: list, action) -> str:
        return "allow"  # for now: allow everything


# ---------- 4. The agent loop ----------
def run_agent(user_request: str) -> str:
    guard = Guard()
    messages = [{"role": "user", "content": user_request}]

    for step in range(1, MAX_STEPS + 1):
        resp = client.messages.create(
            model=MODEL, max_tokens=1024, tools=TOOLS, messages=messages
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            print(f"[step {step}] {block.name}({block.input})")

            decision = guard.check(user_request, messages, block)
            if decision == "allow":
                output, is_error = run_tool(block.name, block.input)
            else:
                output, is_error = f"Action blocked by guard: {decision}", True

            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
                "is_error": is_error,
            })
        messages.append({"role": "user", "content": results})

    return f"Stopped: reached the limit of {MAX_STEPS} steps."


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python agent.py "<your request>"')
        sys.exit(1)
    print(run_agent(sys.argv[1]))
