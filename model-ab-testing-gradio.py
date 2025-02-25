import gradio as gr
import json
import logging
import time
import pandas as pd
import os
import re
import requests
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from tenacity import retry, stop_after_attempt, wait_exponential
import csv
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("model_tester")


@dataclass
class ModelEndpoint:
    """Simple model endpoint configuration."""
    name: str
    api_url: str
    api_key: str
    model_id: str
    max_tokens: int = 1024
    temperature: float = 0.0
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "name": self.name,
            "api_url": self.api_url,
            "model_id": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


@dataclass
class TestCase:
    """Test case containing a key to query and actual value for evaluation."""
    key: str
    value: str
    id: Optional[str] = None


@dataclass
class ModelResponse:
    """Model response for a test case."""
    test_id: str
    model_name: str
    output: str
    latency: float


@dataclass
class EvaluationResult:
    """Evaluation result from the LM judge."""
    test_id: str
    champion_output: str
    challenger_output: str
    winner: str  # "CHAMPION", "CHALLENGER", or "TIE"
    confidence: float
    reasoning: str


class ModelRunner:
    """Handles model API calls."""
    
    def __init__(self, endpoint: ModelEndpoint, prompt_template: str):
        self.endpoint = endpoint
        self.prompt_template = prompt_template
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate(self, test_case: TestCase) -> ModelResponse:
        """Call the model API with the test case."""
        start_time = time.time()
        
        # Format prompt using the key
        prompt = self.prompt_template.format(key=test_case.key)
        
        # Determine API type and make appropriate call
        if "openai" in self.endpoint.api_url.lower():
            response = self._call_openai_api(prompt)
        elif "anthropic" in self.endpoint.api_url.lower():
            response = self._call_anthropic_api(prompt)
        elif "ollama" in self.endpoint.api_url.lower() or ":11434" in self.endpoint.api_url:
            response = self._call_ollama_api(prompt)
        elif "generativelanguage.googleapis.com" in self.endpoint.api_url:
            response = self._call_gemini_api(prompt)
        else:
            response = self._call_generic_api(prompt)
        
        end_time = time.time()
        
        return ModelResponse(
            test_id=test_case.id or test_case.key[:10],
            model_name=self.endpoint.name,
            output=response,
            latency=end_time - start_time,
        )
        
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.endpoint.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": self.endpoint.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.endpoint.max_tokens,
            "temperature": self.endpoint.temperature,
        }
        
        response = requests.post(
            self.endpoint.api_url, 
            headers=headers, 
            json=data
        )
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def _call_anthropic_api(self, prompt: str) -> str:
        """Call Anthropic API."""
        headers = {
            "x-api-key": self.endpoint.api_key,
            "Content-Type": "application/json",
        }
        
        data = {
            "model": self.endpoint.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.endpoint.max_tokens,
            "temperature": self.endpoint.temperature,
        }
        
        response = requests.post(
            self.endpoint.api_url, 
            headers=headers, 
            json=data
        )
        response.raise_for_status()
        
        result = response.json()
        return result["content"][0]["text"]
    
    def _call_ollama_api(self, prompt: str) -> str:
        """Call Ollama API."""
        data = {
            "model": self.endpoint.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.endpoint.temperature,
                "num_predict": self.endpoint.max_tokens
            }
        }
        
        response = requests.post(
            self.endpoint.api_url, 
            json=data
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "")
    
    def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API."""
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.endpoint.api_key:
            url = f"{self.endpoint.api_url}?key={self.endpoint.api_key}"
        else:
            url = self.endpoint.api_url
        
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": self.endpoint.max_tokens,
                "temperature": self.endpoint.temperature
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    
    def _call_generic_api(self, prompt: str) -> str:
        """Call a generic API endpoint."""
        headers = {
            "Authorization": f"Bearer {self.endpoint.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "prompt": prompt,
            "model": self.endpoint.model_id,
            "max_tokens": self.endpoint.max_tokens,
            "temperature": self.endpoint.temperature,
        }
        
        response = requests.post(
            self.endpoint.api_url, 
            headers=headers, 
            json=data
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get("output", str(result))


class LMJudge:
    """Uses a language model to judge between champion and challenger outputs."""
    
    def __init__(
        self, 
        endpoint: ModelEndpoint,
        evaluation_prompt_template: str,
    ):
        self.endpoint = endpoint
        self.evaluation_prompt_template = evaluation_prompt_template
        self.model_runner = ModelRunner(endpoint, "{prompt}")
    
    def evaluate(
        self, 
        test_case: TestCase, 
        champion_response: ModelResponse, 
        challenger_response: ModelResponse
    ) -> EvaluationResult:
        """Evaluate champion vs challenger outputs against the true value."""
        evaluation_prompt = self.evaluation_prompt_template.format(
            key=test_case.key,
            value=test_case.value,
            champion_output=champion_response.output,
            challenger_output=challenger_response.output,
        )
        
        # Get judge's response
        judge_response = self.model_runner.generate(
            TestCase(key=evaluation_prompt, id=f"judge_{test_case.id}")
        )
        
        # Parse the judge's decision
        parsed_result = self._parse_judge_response(judge_response.output)
        
        return EvaluationResult(
            test_id=test_case.id or test_case.key[:10],
            champion_output=champion_response.output,
            challenger_output=challenger_response.output,
            winner=parsed_result["winner"],
            confidence=parsed_result["confidence"],
            reasoning=parsed_result["reasoning"],
        )
    
    def _parse_judge_response(self, response: str) -> Dict[str, Any]:
        """
        Store the full response without attempting complex parsing.
        This allows for simpler Excel post-processing.
        """
        # Store the raw response for Excel post-processing
        return {
            "winner": "UNDETERMINED",  # Will be determined in Excel
            "confidence": 0.0,         # Will be determined in Excel
            "reasoning": response,     # The full response for manual analysis
        }


class ResultAggregator:
    """Collects raw evaluation results for Excel post-processing."""
    
    def __init__(self, confidence_threshold: float = 0.8):
        # Keep parameter for backward compatibility, though not used
        self.confidence_threshold = confidence_threshold
    
    def aggregate(self, evaluation_results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Simply collect all results for Excel post-processing.
        No filtering or counting is done here.
        """
        return {
            "total_evaluations": len(evaluation_results),
            "raw_evaluations": [
                {
                    "test_id": result.test_id,
                    "champion_output": result.champion_output,
                    "challenger_output": result.challenger_output,
                    "judge_response": result.reasoning,
                    # The following fields will be filled in Excel
                    "winner": "TBD IN EXCEL",
                    "confidence": "TBD IN EXCEL"
                } 
                for result in evaluation_results
            ],
            "note": "Results need post-processing in Excel to determine winners and calculate statistics."
        }


class ModelTester:
    """Main class that orchestrates the A/B testing process."""
    
    def __init__(
        self,
        champion_endpoint: ModelEndpoint,
        challenger_endpoint: ModelEndpoint,
        judge_endpoint: ModelEndpoint,
        model_prompt_template: str,
        evaluation_prompt_template: str,
        confidence_threshold: float = 0.8,
    ):
        self.champion_runner = ModelRunner(champion_endpoint, model_prompt_template)
        self.challenger_runner = ModelRunner(challenger_endpoint, model_prompt_template)
        self.judge = LMJudge(judge_endpoint, evaluation_prompt_template)
        self.aggregator = ResultAggregator(confidence_threshold)
        self.champion_endpoint = champion_endpoint
        self.challenger_endpoint = challenger_endpoint
        self.judge_endpoint = judge_endpoint
    
    def run_test(self, test_cases: List[TestCase], batch_size: int = 10, progress=None) -> Dict[str, Any]:
        """Run the complete test process."""
        all_evaluation_results = []
        champion_metrics = {"total_latency": 0, "total_tokens": 0}
        challenger_metrics = {"total_latency": 0, "total_tokens": 0}
        judge_metrics = {"total_latency": 0, "total_tokens": 0}
        
        total_batches = (len(test_cases) + batch_size - 1) // batch_size
        
        # Process in batches
        for i in range(0, len(test_cases), batch_size):
            batch = test_cases[i:i + batch_size]
            batch_num = i // batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            if progress is not None:
                progress(batch_num / total_batches, f"Processing batch {batch_num}/{total_batches}")
            
            # 1. Get responses from both models
            champion_responses = []
            challenger_responses = []
            
            for test_case in batch:
                # Assign an ID if not present
                if not test_case.id:
                    test_case.id = test_case.key[:10]
                
                # Get champion response
                champion_response = self.champion_runner.generate(test_case)
                champion_responses.append(champion_response)
                champion_metrics["total_latency"] += champion_response.latency
                champion_metrics["total_tokens"] += len(champion_response.output.split())
                
                # Get challenger response
                challenger_response = self.challenger_runner.generate(test_case)
                challenger_responses.append(challenger_response)
                challenger_metrics["total_latency"] += challenger_response.latency
                challenger_metrics["total_tokens"] += len(challenger_response.output.split())
            
            # 2. Evaluate with LM judge
            batch_evaluation_results = []
            for idx, test_case in enumerate(batch):
                start_time = time.time()
                evaluation_result = self.judge.evaluate(
                    test_case,
                    champion_responses[idx],
                    challenger_responses[idx],
                )
                judge_latency = time.time() - start_time
                judge_metrics["total_latency"] += judge_latency
                judge_metrics["total_tokens"] += len(evaluation_result.reasoning.split())
                
                batch_evaluation_results.append(evaluation_result)
            
            all_evaluation_results.extend(batch_evaluation_results)
            
            # Extract and log verdicts from this batch
            batch_counts = {"MODEL_A_WINS": 0, "MODEL_B_WINS": 0, "TIE": 0, "UNKNOWN": 0}
            for result in batch_evaluation_results:
                verdict_match = re.search(r"VERDICT:\s*(MODEL_A_WINS|MODEL_B_WINS|TIE)", result.reasoning)
                if verdict_match:
                    verdict = verdict_match.group(1)
                    batch_counts[verdict] += 1
                else:
                    batch_counts["UNKNOWN"] += 1
            
            logger.info(
                f"Batch results: A wins: {batch_counts['MODEL_A_WINS']}, "
                f"B wins: {batch_counts['MODEL_B_WINS']}, "
                f"Ties: {batch_counts['TIE']}, "
                f"Unknown: {batch_counts['UNKNOWN']}"
            )
        
        # Calculate average metrics
        num_cases = len(test_cases)
        champion_metrics["avg_latency"] = champion_metrics["total_latency"] / num_cases
        champion_metrics["avg_tokens"] = champion_metrics["total_tokens"] / num_cases
        challenger_metrics["avg_latency"] = challenger_metrics["total_latency"] / num_cases
        challenger_metrics["avg_tokens"] = challenger_metrics["total_tokens"] / num_cases
        judge_metrics["avg_latency"] = judge_metrics["total_latency"] / num_cases
        judge_metrics["avg_tokens"] = judge_metrics["total_tokens"] / num_cases
        
        # 3. Aggregate results
        aggregated_results = self.aggregator.aggregate(all_evaluation_results)
        
        # 4. Summarize results by counting verdicts
        verdicts = {"MODEL_A_WINS": 0, "MODEL_B_WINS": 0, "TIE": 0, "UNKNOWN": 0}
        confidences = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        
        for result in all_evaluation_results:
            verdict_match = re.search(r"VERDICT:\s*(MODEL_A_WINS|MODEL_B_WINS|TIE)", result.reasoning)
            if verdict_match:
                verdict = verdict_match.group(1)
                verdicts[verdict] += 1
            else:
                verdicts["UNKNOWN"] += 1
            
            confidence_match = re.search(r"CONFIDENCE:\s*(\d+)/5", result.reasoning)
            if confidence_match:
                confidence = confidence_match.group(1)
                confidences[confidence] = confidences.get(confidence, 0) + 1
        
        # Calculate percentages
        total_verdicts = sum(verdicts.values())
        verdict_percentages = {
            k: round(v / total_verdicts * 100, 2) if total_verdicts > 0 else 0
            for k, v in verdicts.items() if k != "UNKNOWN"
        }
        
        # Determine overall winner
        decision = "MAINTAIN_CHAMPION"
        reason = "Challenger did not demonstrate significant improvement"
        
        # Challenger needs to win by a margin to replace champion
        if verdict_percentages.get("MODEL_B_WINS", 0) > 55 and verdict_percentages.get("MODEL_B_WINS", 0) > verdict_percentages.get("MODEL_A_WINS", 0):
            decision = "REPLACE_WITH_CHALLENGER"
            reason = f"Challenger outperformed champion ({verdict_percentages.get('MODEL_B_WINS', 0)}% vs {verdict_percentages.get('MODEL_A_WINS', 0)}%)"
        
        # Log final results
        logger.info(
            f"Final verdict counts: A wins: {verdicts['MODEL_A_WINS']} ({verdict_percentages.get('MODEL_A_WINS', 0)}%), "
            f"B wins: {verdicts['MODEL_B_WINS']} ({verdict_percentages.get('MODEL_B_WINS', 0)}%), "
            f"Ties: {verdicts['TIE']} ({verdict_percentages.get('TIE', 0)}%)"
        )
        logger.info(f"Decision: {decision} - {reason}")
        
        if progress is not None:
            progress(1.0, "Testing completed")
        
        return {
            "evaluations": [result.__dict__ for result in all_evaluation_results],
            "summary": {
                "total_test_cases": num_cases,
                "verdicts": verdicts,
                "verdict_percentages": verdict_percentages,
                "confidences": confidences,
                "decision": decision,
                "reason": reason,
                "champion_metrics": champion_metrics,
                "challenger_metrics": challenger_metrics,
                "judge_metrics": judge_metrics,
                "champion_name": self.champion_endpoint.name,
                "challenger_name": self.challenger_endpoint.name,
                "judge_name": self.judge_endpoint.name,
            }
        }


def save_results_to_json(results: Dict[str, Any], file_path: str):
    """Save results to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)


def save_results_to_csv(results: Dict[str, Any], file_path: str):
    """
    Save evaluation results to a CSV file for Excel processing.
    Uses the standardized verdict format for easier sorting and filtering.
    """
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            "test_id", 
            "query", 
            "champion_model",
            "challenger_model",
            "verdict",  # Automatically extracted from judge response
            "confidence",  # Automatically extracted from judge response
            "explanation",
            "champion_output", 
            "challenger_output", 
            "full_judge_response"
        ])
        
        # Write data rows
        for eval_item in results.get("evaluations", []):
            # Extract verdict and confidence using regex
            verdict = "UNKNOWN"
            confidence = "0"
            judge_response = eval_item.get("reasoning", "")
            
            # Extract verdict
            verdict_match = re.search(r"VERDICT:\s*(MODEL_A_WINS|MODEL_B_WINS|TIE)", judge_response)
            if verdict_match:
                verdict = verdict_match.group(1)
            
            # Extract confidence
            confidence_match = re.search(r"CONFIDENCE:\s*(\d+)/5", judge_response)
            if confidence_match:
                confidence = confidence_match.group(1)
            
            # Extract explanation (everything after the confidence line)
            explanation = ""
            lines = judge_response.split('\n')
            capture = False
            for line in lines:
                if capture:
                    explanation += line + " "
                if "CONFIDENCE:" in line:
                    capture = True
            
            writer.writerow([
                eval_item.get("test_id", ""),
                "",  # query column - to be filled manually if needed
                results.get("summary", {}).get("champion_name", ""),
                results.get("summary", {}).get("challenger_name", ""),
                verdict,
                confidence,
                explanation.strip(),
                eval_item.get("champion_output", "")[:300],  # Truncate long outputs
                eval_item.get("challenger_output", "")[:300],  # Truncate long outputs
                judge_response
            ])


def load_test_cases_from_csv_url(csv_url: str, key_column: str = "text", value_column: str = "label", 
                              limit: int = None) -> List[TestCase]:
    """Load test cases from a CSV URL."""
    try:
        # Download the CSV file
        response = requests.get(csv_url)
        response.raise_for_status()
        
        # Parse CSV content
        csv_content = response.content.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_content))
        
        # Validate columns
        if key_column not in df.columns:
            raise ValueError(f"Key column '{key_column}' not found in CSV. Available columns: {', '.join(df.columns)}")
        
        if value_column not in df.columns:
            raise ValueError(f"Value column '{value_column}' not found in CSV. Available columns: {', '.join(df.columns)}")
        
        # Create test cases
        test_cases = []
        for i, row in df.iterrows():
            if limit is not None and i >= limit:
                break
                
            test_cases.append(TestCase(
                key=str(row[key_column]),
                value=str(row[value_column]),
                id=f"case_{i}",
            ))
        
        return test_cases
    
    except Exception as e:
        logger.error(f"Error loading CSV from URL {csv_url}: {str(e)}")
        raise


def create_model_endpoint(model_type, model_name, api_url, api_key, model_id, max_tokens, temperature):
    """Create a model endpoint configuration based on provided parameters."""
    return ModelEndpoint(
        name=model_name,
        api_url=api_url,
        api_key=api_key,
        model_id=model_id,
        max_tokens=int(max_tokens),
        temperature=float(temperature)
    )


def run_model_test(
    champion_type, champion_name, champion_api_url, champion_api_key, champion_model_id, champion_max_tokens, champion_temperature,
    challenger_type, challenger_name, challenger_api_url, challenger_api_key, challenger_model_id, challenger_max_tokens, challenger_temperature,
    judge_type, judge_name, judge_api_url, judge_api_key, judge_model_id, judge_max_tokens, judge_temperature,
    csv_url, key_column, value_column, limit, output_dir, prompt_template, evaluation_template,
    progress=gr.Progress()
):
    """Run a model test with the provided parameters and return results."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create model endpoints
        champion_endpoint = create_model_endpoint(
            champion_type, champion_name, champion_api_url, champion_api_key, champion_model_id,
            champion_max_tokens, champion_temperature
        )
        
        challenger_endpoint = create_model_endpoint(
            challenger_type, challenger_name, challenger_api_url, challenger_api_key, challenger_model_id,
            challenger_max_tokens, challenger_temperature
        )
        
        judge_endpoint = create_model_endpoint(
            judge_type, judge_name, judge_api_url, judge_api_key, judge_model_id,
            judge_max_tokens, judge_temperature
        )
        
        # Load test cases
        progress(0.1, "Loading test cases from CSV")
        test_cases = load_test_cases_from_csv_url(
            csv_url, 
            key_column, 
            value_column, 
            int(limit) if limit else None
        )
        
        # Set up the model tester
        tester = ModelTester(
            champion_endpoint=champion_endpoint,
            challenger_endpoint=challenger_endpoint,
            judge_endpoint=judge_endpoint,
            model_prompt_template=prompt_template,
            evaluation_prompt_template=evaluation_template,
        )
        
        # Run the test
        progress(0.2, f"Starting test with {len(test_cases)} test cases")
        results = tester.run_test(test_cases, batch_size=5, progress=progress)
        
        # Save results
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        json_path = os.path.join(output_dir, f"results-{timestamp}.json")
        csv_path = os.path.join(output_dir, f"results-{timestamp}.csv")
        
        save_results_to_json(results, json_path)
        save_results_to_csv(results, csv_path)
        
        # Format summary for display
        summary = results["summary"]
        verdicts = summary["verdicts"]
        verdict_percentages = summary["verdict_percentages"]
        
        summary_text = f"""
## Test Results Summary

**Models:**
- Champion: {summary['champion_name']}
- Challenger: {summary['challenger_name']}
- Judge: {summary['judge_name']}

**Verdict Counts:**
- MODEL A WINS: {verdicts['MODEL_A_WINS']} ({verdict_percentages.get('MODEL_A_WINS', 0)}%)
- MODEL B WINS: {verdicts['MODEL_B_WINS']} ({verdict_percentages.get('MODEL_B_WINS', 0)}%)
- TIE: {verdicts['TIE']} ({verdict_percentages.get('TIE', 0)}%)
- UNKNOWN: {verdicts['UNKNOWN']}

**Decision:** {summary['decision']}
**Reason:** {summary['reason']}

**Performance Metrics:**
- Champion Avg Latency: {summary['champion_metrics']['avg_latency']:.2f}s
- Challenger Avg Latency: {summary['challenger_metrics']['avg_latency']:.2f}s
- Judge Avg Latency: {summary['judge_metrics']['avg_latency']:.2f}s

**Files Saved:**
- JSON: {json_path}
- CSV: {csv_path}
"""
        
        return summary_text, json_path, csv_path
    
    except Exception as e:
        logger.exception("Error running model test")
        return f"Error: {str(e)}", None, None


# Default templates
DEFAULT_PROMPT_TEMPLATE = """
Please provide information about the following topic: {key}

Be concise yet comprehensive and factually accurate.
"""

DEFAULT_EVALUATION_TEMPLATE = """
I need you to evaluate two model responses about a topic, comparing them with the reference value.

Topic: {key}

Reference value: {value}

Response from Model A (Champion):
```
{champion_output}
```

Response from Model B (Challenger):
```
{challenger_output}
```

Please carefully evaluate both responses in terms of:
1. Factual accuracy compared to the reference value
2. Comprehensiveness of information
3. Clarity and conciseness

IMPORTANT: Your response MUST start with exactly ONE of these specific phrases:
- "VERDICT: MODEL_A_WINS" (if Model A is better)
- "VERDICT: MODEL_B_WINS" (if Model B is better) 
- "VERDICT: TIE" (if both are equally good)

Then on a new line, add:
- "CONFIDENCE: X/5" (where X is a number from 1-5, with 5 being highest confidence)

After these two required lines, explain your reasoning briefly.
"""

# Gradio interface
def create_gradio_interface():
    """Create a Gradio interface for the model tester."""
    # CSS for better layout
    css = """
    .container { max-width: 1200px; margin: auto; }
    .model-group { border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px; }
    .model-group h3 { margin-top: 0; }
    """
    
    with gr.Blocks(css=css) as app:
        gr.Markdown("# Model A/B Testing Tool")
        gr.Markdown("Compare two models against a reference value using a judge model")
        
        with gr.Tab("Test Configuration"):
            with gr.Group(elem_classes="model-group"):
                gr.Markdown("### Champion Model (A)")
                with gr.Row():
                    champion_type = gr.Dropdown(
                        choices=["OpenAI", "Anthropic", "Gemini", "Ollama", "Generic"], 
                        value="OpenAI", 
                        label="Provider"
                    )
                    champion_name = gr.Textbox(label="Model Name", value="Champion-GPT-3.5")
                
                with gr.Row():
                    champion_api_url = gr.Textbox(
                        label="API URL", 
                        value="https://api.openai.com/v1/chat/completions"
                    )
                    champion_api_key = gr.Textbox(
                        label="API Key", 
                        value="", 
                        type="password"
                    )
                
                with gr.Row():
                    champion_model_id = gr.Textbox(label="Model ID", value="gpt-3.5-turbo")
                    champion_max_tokens = gr.Number(label="Max Tokens", value=1024, minimum=1)
                    champion_temperature = gr.Slider(
                        label="Temperature", 
                        minimum=0.0, 
                        maximum=1.0, 
                        value=0.0, 
                        step=0.1
                    )
            
            with gr.Group(elem_classes="model-group"):
                gr.Markdown("### Challenger Model (B)")
                with gr.Row():
                    challenger_type = gr.Dropdown(
                        choices=["OpenAI", "Anthropic", "Gemini", "Ollama", "Generic"], 
                        value="Anthropic", 
                        label="Provider"
                    )
                    challenger_name = gr.Textbox(label="Model Name", value="Challenger-Claude")
                
                with gr.Row():
                    challenger_api_url = gr.Textbox(
                        label="API URL", 
                        value="https://api.anthropic.com/v1/messages"
                    )
                    challenger_api_key = gr.Textbox(
                        label="API Key", 
                        value="", 
                        type="password"
                    )
                
                with gr.Row():
                    challenger_model_id = gr.Textbox(label="Model ID", value="claude-3-haiku-20240307")
                    challenger_max_tokens = gr.Number(label="Max Tokens", value=1024, minimum=1)
                    challenger_temperature = gr.Slider(
                        label="Temperature", 
                        minimum=0.0, 
                        maximum=1.0, 
                        value=0.0, 
                        step=0.1
                    )
            
            with gr.Group(elem_classes="model-group"):
                gr.Markdown("### Judge Model (C)")
                with gr.Row():
                    judge_type = gr.Dropdown(
                        choices=["OpenAI", "Anthropic", "Gemini", "Ollama", "Generic"], 
                        value="OpenAI", 
                        label="Provider"
                    )
                    judge_name = gr.Textbox(label="Model Name", value="Judge-GPT-4")
                
                with gr.Row():
                    judge_api_url = gr.Textbox(
                        label="API URL", 
                        value="https://api.openai.com/v1/chat/completions"
                    )
                    judge_api_key = gr.Textbox(
                        label="API Key", 
                        value="", 
                        type="password"
                    )
                
                with gr.Row():
                    judge_model_id = gr.Textbox(label="Model ID", value="gpt-4")
                    judge_max_tokens = gr.Number(label="Max Tokens", value=2048, minimum=1)
                    judge_temperature = gr.Slider(
                        label="Temperature", 
                        minimum=0.0, 
                        maximum=1.0, 
                        value=0.0, 
                        step=0.1
                    )
            
            with gr.Group(elem_classes="model-group"):
                gr.Markdown("### Data Source & Test Settings")
                with gr.Row():
                    csv_url = gr.Textbox(
                        label="CSV URL", 
                        value="https://example.com/data.csv", 
                        info="URL to CSV file with test cases"
                    )
                
                with gr.Row():
                    key_column = gr.Textbox(
                        label="Key Column", 
                        value="text", 
                        info="Column name for input data (queries)"
                    )
                    value_column = gr.Textbox(
                        label="Value Column", 
                        value="label", 
                        info="Column name for reference/ground truth"
                    )
                    limit = gr.Number(
                        label="Limit", 
                        value=10, 
                        info="Maximum number of test cases to use (0 for all)"
                    )
                
                with gr.Row():
                    output_dir = gr.Textbox(
                        label="Output Directory", 
                        value="./results", 
                        info="Directory to save results"
                    )
            
            with gr.Group(elem_classes="model-group"):
                gr.Markdown("### Prompt Templates")
                with gr.Tabs():
                    with gr.TabItem("Model Prompt"):
                        prompt_template = gr.Textbox(
                            label="Model Prompt Template", 
                            value=DEFAULT_PROMPT_TEMPLATE,
                            lines=10,
                            info="Use {key} as placeholder for the input"
                        )
                    
                    with gr.TabItem("Evaluation Prompt"):
                        evaluation_template = gr.Textbox(
                            label="Evaluation Prompt Template", 
                            value=DEFAULT_EVALUATION_TEMPLATE,
                            lines=15,
                            info="Use {key}, {value}, {champion_output}, and {challenger_output} as placeholders"
                        )
            
            run_button = gr.Button("Run Test", variant="primary")
        
        with gr.Tab("Results"):
            with gr.Group():
                results_markdown = gr.Markdown("Results will appear here after running the test")
                
                with gr.Row():
                    json_file = gr.File(label="JSON Results", interactive=False)
                    csv_file = gr.File(label="CSV Results", interactive=False)
        
        # Define the function to run when the button is clicked
        run_button.click(
            fn=run_model_test,
            inputs=[
                champion_type, champion_name, champion_api_url, champion_api_key, champion_model_id, champion_max_tokens, champion_temperature,
                challenger_type, challenger_name, challenger_api_url, challenger_api_key, challenger_model_id, challenger_max_tokens, challenger_temperature,
                judge_type, judge_name, judge_api_url, judge_api_key, judge_model_id, judge_max_tokens, judge_temperature,
                csv_url, key_column, value_column, limit, output_dir, prompt_template, evaluation_template
            ],
            outputs=[results_markdown, json_file, csv_file]
        )
        
        # Add example data
        gr.Examples(
            examples=[
                ["OpenAI", "GPT-3.5", "https://api.openai.com/v1/chat/completions", "YOUR_KEY", "gpt-3.5-turbo", 1024, 0.0,
                 "Anthropic", "Claude-Haiku", "https://api.anthropic.com/v1/messages", "YOUR_KEY", "claude-3-haiku-20240307", 1024, 0.0,
                 "OpenAI", "GPT-4", "https://api.openai.com/v1/chat/completions", "YOUR_KEY", "gpt-4", 2048, 0.0,
                 "https://example.com/data.csv", "text", "label", 10, "./results"],
                ["Ollama", "Llama3-8B", "http://localhost:11434/api/generate", "", "llama3:8b", 1024, 0.0,
                 "Gemini", "Gemini-Pro", "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent", "YOUR_KEY", "gemini-pro", 1024, 0.0,
                 "Anthropic", "Claude-Opus", "https://api.anthropic.com/v1/messages", "YOUR_KEY", "claude-3-opus-20240229", 2048, 0.0,
                 "https://example.com/data.csv", "text", "label", 5, "./results"]
            ],
            inputs=[
                champion_type, champion_name, champion_api_url, champion_api_key, champion_model_id, champion_max_tokens, champion_temperature,
                challenger_type, challenger_name, challenger_api_url, challenger_api_key, challenger_model_id, challenger_max_tokens, challenger_temperature,
                judge_type, judge_name, judge_api_url, judge_api_key, judge_model_id, judge_max_tokens, judge_temperature,
                csv_url, key_column, value_column, limit, output_dir
            ]
        )
    
    return app


# Run the Gradio app
if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch()
