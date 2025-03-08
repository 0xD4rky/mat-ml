import time
import ast
import re
import io
import sys
import json
import subprocess
from contextlib import redirect_stdout
from datasets import load_dataset
from bespokelabs import curator

def extract_code(solution):
    """Extract code from markdown code blocks or return as is"""
    solution = solution.strip()
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(solution)
    if matches:
        return matches[-1].strip()
    elif solution.startswith("```python"):
        solution = solution[len("```python"):].lstrip()
        if solution.endswith("```"):
            solution = solution[:-3].rstrip()

    return solution

class PrimeIntellectExecutor(curator.CodeExecutor):
    def code(self, row):
        solution = row.get("gold_standard_solution", "")
        if not isinstance(solution, str) or not solution.strip():
            return "# No solution provided"
        
        solution = extract_code(solution)
        
        verification_info = row.get("verification_info", "")
        try:
            verification_dict = ast.literal_eval(verification_info)
            test_cases = verification_dict.get("test_cases", [])
        except Exception:
            test_cases = []
            try:
                input_match = re.search(r"'input'\s*:\s*'((?:\\.|[^\\'])*)'", verification_info, re.DOTALL)
                output_match = re.search(r"'output'\s*:\s*'((?:\\.|[^\\'])*)'", verification_info, re.DOTALL)
                if input_match and output_match:
                    input_text = bytes(input_match.group(1), "utf-8").decode("unicode_escape")
                    output_text = bytes(output_match.group(1), "utf-8").decode("unicode_escape")
                    test_cases = [{"type": "stdin_stdout", "input": input_text, "output": output_text}]
            except Exception:
                pass
        
        test_cases_json = json.dumps(test_cases)