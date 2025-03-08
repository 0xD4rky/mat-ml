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

        wrapper_code = f'''
import sys
import tempfile
import subprocess
import os
import json

# Create a temporary file with the solution code
with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
    f.write("""
{solution}
""")
    temp_file = f.name

try:
    # Create a test case handler
    def run_test_cases(test_cases_json):
        test_cases = json.loads(test_cases_json)
        passed = 0
        total = len(test_cases)
        results = []
        
        for i, case in enumerate(test_cases):
            if case.get("type") != "stdin_stdout":
                continue
                
            test_input = case.get("input", "")
            expected_output = case.get("output", "")
            
            try:
                # Run the temporary file with the test input
                process = subprocess.run(
                    ["python3", temp_file],
                    input=test_input,
                    text=True,
                    capture_output=True,
                    timeout=10  # Set a reasonable timeout
                )
                
                if process.returncode != 0:  # Error in execution
                    results.append({{
                        "test": i + 1,
                        "passed": False,
                        "error": process.stderr,
                        "actual": process.stdout,
                        "expected": expected_output
                    }})
                    continue
                
                # Get output and normalize
                actual_output = process.stdout
                
                # Normalize outputs (strip trailing whitespace)
                actual_lines = [line.rstrip() for line in actual_output.strip().split('\\n')]
                expected_lines = [line.rstrip() for line in expected_output.strip().split('\\n')]
                
                # Remove empty lines at the end if any
                while actual_lines and not actual_lines[-1]:
                    actual_lines.pop()
                while expected_lines and not expected_lines[-1]:
                    expected_lines.pop()
                
                normalized_actual = '\\n'.join(actual_lines)
                normalized_expected = '\\n'.join(expected_lines)
                
                # Check if normalized outputs match
                if normalized_actual == normalized_expected:
                    passed += 1
                    results.append({{
                        "test": i + 1,
                        "passed": True,
                        "actual": actual_output,
                        "expected": expected_output
                    }})
                else:
                    results.append({{
                        "test": i + 1,
                        "passed": False,
                        "actual": actual_output,
                        "expected": expected_output,
                        "normalized_actual": normalized_actual,
                        "normalized_expected": normalized_expected
                    }})
                
            except Exception as e:
                results.append({{
                    "test": i + 1,
                    "passed": False,
                    "error": str(e),
                    "expected": expected_output
                }})
        
        return {{
            "passed": passed,
            "total": total,
            "success_rate": (passed / total) if total > 0 else 0,
            "results": results
        }}
    
    # Define test cases
    test_cases_json = '{test_cases_json}'
    
    # Run tests
    test_results = run_test_cases(test_cases_json)
    
    # Print results for the executor to capture
    print(json.dumps(test_results), end="")
    
    # Also print to stderr so we can see in the log
    if test_results["passed"] == test_results["total"]:
        print("\\n== ALL TESTS PASSED ==", file=sys.stderr)
    else:
        print(f"\\n== {{test_results['passed']}}/{{test_results['total']}} TESTS PASSED ==", file=sys.stderr)
    
    # Print detailed results
    for result in test_results["results"]:
        if not result["passed"]:
            print(f"Test {{result['test']}} failed:", file=sys.stderr)
            print(f"Expected: {{repr(result['expected'])}}", file=sys.stderr)
            print(f"Got: {{repr(result['actual'])}}", file=sys.stderr)

except Exception as e:
    print(json.dumps({{"error": str(e)}}), end="")
    print(f"Execution error: {{e}}", file=sys.stderr)

finally:
    # Clean up the temporary file
    try:
        os.unlink(temp_file)
    except Exception:
        pass
'''