import json
import ollama

MODEL = "qwen2.5:7b"


class BaseAgent:
    """Lightweight reusable LLM agent with isolated system prompt and tool registry."""

    def __init__(self, name: str, system_prompt: str, tools: list, tool_map: dict, max_steps: int = 6):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self.tool_map = tool_map
        self.max_steps = max_steps

    def run(self, task: str) -> str:
        print(f"\n>>> [{self.name}] Task: {task}")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]

        for _ in range(self.max_steps):
            response = ollama.chat(model=MODEL, messages=messages, tools=self.tools)
            msg = response["message"]
            messages.append(msg)

            if not msg.get("tool_calls"):
                print(f"<<< [{self.name}] Done.")
                return msg["content"]

            for call in msg["tool_calls"]:
                fn_name = call["function"]["name"]
                fn_args = call["function"]["arguments"]
                print(f"  [{self.name} -> Tool] {fn_name}({json.dumps(fn_args)})")

                if fn_name not in self.tool_map:
                    result = {"error": f"Tool '{fn_name}' is not permitted for {self.name}."}
                else:
                    try:
                        result = self.tool_map[fn_name](**fn_args)
                    except Exception as e:
                        result = {"error": str(e)}

                messages.append({"role": "tool", "content": json.dumps(result)})

        return f"[{self.name}] Reached max_steps ({self.max_steps})."