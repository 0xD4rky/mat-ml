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
        
        return wrapper_code

    def code_input(self, row):
        """Legacy method for compatibility"""
        verification_info = row.get("verification_info", "")
        if not verification_info:
            return ""
        try:
            verification_dict = ast.literal_eval(verification_info)
            test_cases = verification_dict.get("test_cases", [])
            for test_case in test_cases:
                if test_case.get("type") == "stdin_stdout":
                    test_input = test_case.get("input", "")
                    return bytes(test_input, "utf-8").decode("unicode_escape")
            return ""
        except Exception:
            input_match = re.search(r"'input'\s*:\s*'((?:\\.|[^\\'])*)'", verification_info, re.DOTALL)
            if input_match:
                return bytes(input_match.group(1), "utf-8").decode("unicode_escape")
            return ""

    def _get_expected_output(self, row):
        """Legacy method for compatibility"""
        verification_info = row.get("verification_info", "")
        if not verification_info:
            return ""
        try:
            verification_dict = ast.literal_eval(verification_info)
            test_cases = verification_dict.get("test_cases", [])
            for test_case in test_cases:
                if test_case.get("type") == "stdin_stdout":
                    return bytes(test_case.get("output", ""), "utf-8").decode("unicode_escape")
            return ""
        except Exception:
            output_match = re.search(r"'output'\s*:\s*'((?:\\.|[^\\'])*)'", verification_info, re.DOTALL)
            if output_match:
                return bytes(output_match.group(1), "utf-8").decode("unicode_escape")
            return ""
        
        def code_output(self, row, execution_output):
        row["execution_stdout"] = execution_output.stdout
        row["execution_stderr"] = execution_output.stderr
        
        try:
            if execution_output.stdout is None:
                row["correct"] = False
                row["error"] = "No output received from execution"
                return row
                
            test_results = json.loads(execution_output.stdout)
            
            if "success_rate" in test_results:
                row["correct"] = test_results["success_rate"] >= 1.0
                row["test_results"] = test_results
                
                for result in test_results.get("results", []):
                    if not result.get("passed"):
                        if "normalized_expected" not in row:
                            row["normalized_expected"] = result.get("normalized_expected", "")
                            row["normalized_actual"] = result.get("normalized_actual", "")
                        if "expected_output" not in row:
                            row["expected_output"] = result.get("expected", "")
            elif "error" in test_results:
                row["correct"] = False
                row["error"] = test_results["error"]
            else:
                row["correct"] = False
                
            return row
        except json.JSONDecodeError:
            # Keep existing fallback logic
            pass


        stderr = execution_output.stderr or ""
        if "== ALL TESTS PASSED ==" in stderr:
            row["correct"] = True
        else:
            try:
                actual_output = execution_output.stdout
                expected_output = self._get_expected_output(row)
                
                actual_lines = [line.rstrip() for line in actual_output.strip().split('\n')]
                expected_lines = [line.rstrip() for line in expected_output.strip().split('\n')]
                
                while actual_lines and not actual_lines[-1]:
                    actual_lines.pop()
                while expected_lines and not expected_lines[-1]:
                    expected_lines.pop()
                    
                normalized_actual = '\n'.join(actual_lines)
                normalized_expected = '\n'.join(expected_lines)
                
                row["correct"] = normalized_actual == normalized_expected
                if not row["correct"]:
                    row["normalized_actual"] = normalized_actual
                    row["normalized_expected"] = normalized_expected
            except Exception as e:
                row["correct"] = False
                row["normalization_error"] = str(e)
                
        try:
            row["expected_output"] = self._get_expected_output(row)
        except Exception:
            pass

        return row

if __name__ == "__main__":
    executor = PrimeIntellectExecutor()
    dataset = load_dataset("PrimeIntellect/verifiable-coding-problems", split="train")
    
    num_samples = min(5, len(dataset))
    selected_data = dataset.select(range(num_samples))
    
    print(f"Testing {num_samples} samples from PrimeIntellect/verifiable-coding-problems")
    
    print("\nDEBUG INFO FOR FIRST PROBLEM:")
    print("Solution code:")
    first_solution = selected_data[0].get("gold_standard_solution", "")
    print(extract_code(first_solution)[:200] + "..." if len(extract_code(first_solution)) > 200 else extract_code(first_solution))
    
    print("\nVerification info:")
    verification_info = selected_data[0].get("verification_info", "")
    print(verification_info[:200] + "..." if len(verification_info) > 200 else verification_info)
    
    if 'verification_info' in selected_data[0]:
        print("\nActual input for first problem:")
        input_text = executor.code_input(selected_data[0])
        print(repr(input_text))
        print("\nExpected output for first problem:")
        expected_output = executor._get_expected_output(selected_data[0])
        print(repr(expected_output))

    try:
        start_time = time.time()
        execution_output = executor(selected_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        correct_count = sum(1 for x in execution_output['correct'] if x)
        print(f"\nSummary: {correct_count}/{num_samples} solutions passed verification")
        print(f"Success rate: {correct_count/num_samples*100:.2f}%")
        print(f"Average execution time: {execution_time/num_samples:.4f} seconds per problem")
        
        for i in range(num_samples):
            print(f"\n--- Problem {i+1}: {selected_data[i].get('problem_id', f'Unknown-{i}')} ---")
            
            is_correct = False
            if 'correct' in execution_output:
                try:
                    is_correct = bool(execution_output['correct'][i])
                except (IndexError, TypeError):
                    pass
            
            print(f"Correct: {is_correct}")
            
            if 'test_results' in execution_output:
                try:
                    test_results = execution_output['test_results'][i]
                    if isinstance(test_results, dict):
                        passed = test_results.get('passed', 0)
                        total = test_results.get('total', 0)
                        print(f"Tests passed: {passed}/{total}")
                except (IndexError, TypeError, AttributeError):
                    print("Could not access test results")
            
            if not is_correct:
                print("\nRaw outputs:")
                
                print("Expected:")
                try:
                    if 'expected_output' in execution_output:
                        expected = execution_output['expected_output'][i]
                        print(repr(expected if expected is not None else ""))
                    else:
                        print("No expected output available")
                except (IndexError, TypeError):
                    print("Could not access expected output")
                
                print("Actual:")
                try:
                    actual = execution_output['execution_stdout'][i]
                    print(repr(actual if actual is not None else ""))
                except (IndexError, TypeError):
                    print("Could not access actual output")
                
                try:
                    if ('normalized_expected' in execution_output and 
                        'normalized_actual' in execution_output):
                        print("\nNormalized comparison:")
                        print("Expected (normalized):")
                        print(repr(execution_output['normalized_expected'][i]))
                        print("Actual (normalized):")
                        print(repr(execution_output['normalized_actual'][i]))
                except (IndexError, TypeError):
                    pass
                
                try:
                    if 'execution_stderr' in execution_output and execution_output['execution_stderr'][i]:
                        print("\nError info:")
                        print(execution_output['execution_stderr'][i])
                except (IndexError, TypeError):
                    pass
                
                try:
                    if 'error' in execution_output:
                        print("Error:")
                        print(execution_output['error'][i])
                except (IndexError, TypeError):
                    pass
    
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()