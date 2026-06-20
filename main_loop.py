from openai import OpenAI
import subprocess
import os
from os import getenv

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=getenv("OPENROUTER_API_KEY"),
)

SYSTEM = """You are a coding agent. Given a task:
1. Write Python code to solve it inside ```python blocks
2. I'll run it and show you the output
3. Fix errors and retry
4. When it works, say DONE"""

def run_code(code: str) -> str:
    with open("temp.py", "w") as f:
        f.write(code)
    result = subprocess.run(
        ["python", "temp.py"],
        capture_output=True, text=True, timeout=10
    )
    return result.stdout + result.stderr

def extract_code(text: str) -> str | None:
    if "```python" not in text:
        return None
    return text.split("```python")[1].split("```")[0].strip()

def run_agent(task: str):
    messages = [{"role": "user", "content": task}]

    for i in range(10):
        response = client.chat.completions.create(
            model="google/gemma-4-26b-a4b-it:free",
            system=SYSTEM,
            messages=messages
        )
        reply = response.content[0].text
        print(f"\n[Agent turn {i+1}]\n{reply}")
        messages.append({"role": "assistant", "content": reply})

        if "DONE" in reply:
            print("\nTask complete.")
            break

        code = extract_code(reply)
        if code:
            output = run_code(code)
            print(f"\n[Output]\n{output}")
            messages.append({"role": "user", "content": f"Output:\n{output}"})
        else:
            messages.append({"role": "user", "content": "No code block found. Write the code."})

if __name__ == "__main__":
    task = input("Task: ")
    run_agent(task)