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
