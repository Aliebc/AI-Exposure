import re
import json
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
import os


# Configuration - will be loaded from environment variables
API_KEY = os.getenv("METACHAT_GEMINI_API_KEY")
MODEL_ID = os.getenv("METACHAT_MODEL_ID", "gemini-2.5-flash")
BASE_URL = os.getenv("METACHAT_BASE_URL", "https://llm-api.mmchat.xyz")
MAX_WORKERS = int(os.getenv("DWA_MAX_WORKERS", "3"))
BASE_DELAY = float(os.getenv("DWA_BASE_DELAY", "1.5"))
MAX_RETRIES = int(os.getenv("DWA_MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.getenv("DWA_REQUEST_TIMEOUT", "120"))


def get_ai_exposure_score(description: str) -> dict:
    """
    Assess AI exposure risk for a work activity description using MetaChat Gemini API.

    Implements exponential backoff retry mechanism for handling rate limit (429) errors.

    Args:
        description (str): The work activity description to assess

    Returns:
        dict: Assessment results containing score, exposure level, and reason
    """
    # Risk classification criteria based on AI exposure research
    system_prompt = """
    You are a professional risk assessment expert. Please strictly evaluate the potential 
    for work activities to be automated by AI according to the following rules:
    
    1. Risk classification standards:
       - E0 (Low risk): 0-30 points, very difficult to replace (requires high creativity/emotional interaction/unstructured decision making)
       - E1 (Medium risk): 31-70 points, partially replaceable (some automation possible)
       - E2 (High risk): 71-100 points, easily replaceable (highly repetitive/rule-based/data-driven)
    
    2. Scoring basis:
       - Physical operation/emotional interaction (score reduction)
       - Rule-based data processing (score increase)
       - Creativity/non-standardized decision making (score reduction)
    
    3. Output requirements:
       - Return JSON format: {"score": number, "exposure": "E0/E1/E2", "reason": "assessment reason"}
       - Score range: 0-100 integer
       - Reason should not exceed 30 characters
    """

    result = {
        "score": np.nan,
        "exposure": "ERROR",
        "reason": "Initialization error",
        "dwa_title": description
    }

    retries = 0
    last_exception = None

    # Check if API key is configured
    if not API_KEY:
        result["reason"] = "MetaChat Gemini API key not configured"
        return result

    # Configure Gemini client
    try:
        genai.configure(
            api_key=API_KEY,
            transport='rest',
            client_options={"api_endpoint": BASE_URL}
        )
    except Exception as e:
        result["reason"] = f"Gemini client configuration failed: {str(e)}"
        return result

    while retries <= MAX_RETRIES:
        try:
            model = genai.GenerativeModel(MODEL_ID)

            full_prompt = f"{system_prompt.strip()}\n\nPlease assess the following work activity: {description}"

            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=200,
                    temperature=0.1,
                )
            )

            content = response.text

            # Try direct JSON parsing first
            try:
                json_data = json.loads(content)
                if all(key in json_data for key in ["score", "exposure", "reason"]):
                    score = int(json_data["score"])
                    exposure = json_data["exposure"]
                    reason = json_data["reason"][:100]  # Truncate overly long reasons

                    # Validate score range
                    if 0 <= score <= 100:
                        result.update({
                            "score": score,
                            "exposure": exposure,
                            "reason": reason
                        })
                        return result
            except json.JSONDecodeError:
                pass  # JSON parsing failed, try regex extraction

            # Fallback: regex extraction
            score_match = re.search(r'"score":\s*(\d+)', content)
            exposure_match = re.search(r'"exposure":\s*"([E0-2]{2})"', content)
            reason_match = re.search(r'"reason":\s*"([^"]+)"', content)

            if score_match and exposure_match and reason_match:
                score = int(score_match.group(1))
                exposure = exposure_match.group(1)
                reason = reason_match.group(1)[:100]

                if 0 <= score <= 100 and exposure in ["E0", "E1", "E2"]:
                    result.update({
                        "score": score,
                        "exposure": exposure,
                        "reason": reason
                    })
                else:
                    result["reason"] = "Invalid score or exposure level in response"
            else:
                result["reason"] = "Failed to extract assessment from response"

            return result

        except Exception as e:
            last_exception = e
            error_msg = str(e)

            # Check for rate limit error (429)
            if ("429" in error_msg or "Too Many Requests" in error_msg or
                "quota" in error_msg.lower()):
                retries += 1
                if retries > MAX_RETRIES:
                    result["reason"] = f"Exceeded maximum retry attempts, still encountering rate limit: {error_msg}"
                    return result

                # Exponential backoff strategy
                wait_time = BASE_DELAY * (2 ** (retries - 1)) * random.uniform(0.8, 1.2)
                print(f"Encountered rate limit (429), retry {retries}/{MAX_RETRIES}, waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                # For other non-429 errors, return immediately
                result["reason"] = f"API call failed: {error_msg}"
                time.sleep(1)  # Brief pause on error
                return result

    # All retries failed
    result["reason"] = f"All retry attempts failed, last error: {str(last_exception)}"
    return result


def parse_dwa_line(line: str) -> dict:
    """
    Parse a line of DWA data to extract dwa_id and dwa_title.

    Args:
        line (str): A line of text from the DWA reference file

    Returns:
        dict: Parsed data containing dwa_id and dwa_title, or None if parsing fails
    """
    try:
        parts = line.strip().split()
        dwa_id = None

        # Find the DWA ID (starts with '4.')
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].startswith('4.'):
                dwa_id = parts[i]
                break

        if not dwa_id:
            return None

        # Extract the DWA title
        if dwa_id in parts:
            index = parts.index(dwa_id)
            dwa_title = " ".join(parts[index + 1:]) if index + 1 < len(parts) else ""
        else:
            dwa_title = " ".join(parts[-1:])

        dwa_title = dwa_title.strip('"').strip()
        return {"dwa_id": dwa_id, "dwa_title": dwa_title}

    except Exception as e:
        print(f"Line parsing error: {str(e)}\nLine content: {line}")
        return None


def load_dwa_data(file_path: str = "DWA_Reference.csv") -> pd.DataFrame:
    """
    Load DWA data from a CSV file.

    Args:
        file_path (str): Path to the DWA reference CSV file

    Returns:
        pd.DataFrame: DataFrame containing DWA IDs and titles
    """
    data = []
    success_count = 0
    failed_lines = []

    # Try different encodings to read the file
    encodings = ['utf-8', 'latin-1', 'iso-8859-1']
    lines = None

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"File reading failed with {encoding}: {e}")
            continue

    if lines is None:
        print("Failed to read file with any encoding")
        return pd.DataFrame()

    # Skip header if present
    start_index = 1 if len(lines) > 0 and "DWA" in lines[0] else 0

    # Parse each line
    for i in range(start_index, len(lines)):
        line = lines[i].strip()
        if not line:
            continue

        parsed = parse_dwa_line(line)
        if parsed and parsed["dwa_title"]:
            data.append(parsed)
            success_count += 1
        else:
            failed_lines.append(i + 1)

    print(f"Successfully parsed {success_count} DWA records")
    if failed_lines:
        print(f"Failed to parse {len(failed_lines)} lines")

    return pd.DataFrame(data)


def evaluate_dwa_concurrent(data: pd.DataFrame, max_workers: int = MAX_WORKERS) -> pd.DataFrame:
    """
    Evaluate DWA data concurrently using thread pooling.

    Args:
        data (pd.DataFrame): DataFrame containing DWA data to evaluate
        max_workers (int): Maximum number of concurrent workers

    Returns:
        pd.DataFrame: Evaluation results
    """
    if data.empty:
        print("Error: No DWA data to evaluate")
        return pd.DataFrame()

    results = []
    futures = []

    # Progress bar for monitoring
    pbar = tqdm(total=len(data), desc="AI Evaluation Progress")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        for idx, row in data.iterrows():
            future = executor.submit(get_ai_exposure_score, row["dwa_title"])
            futures.append((row["dwa_id"], row["dwa_title"], future, time.time()))
            time.sleep(BASE_DELAY / max_workers)  # Rate limiting

        # Process results as they complete
        for dwa_id, dwa_title, future, start_time in futures:
            try:
                result = future.result(timeout=REQUEST_TIMEOUT)
                result.update({
                    "dwa_id": dwa_id,
                    "dwa_title": dwa_title
                })
                results.append(result)

                # Update progress bar with timing info
                duration = time.time() - start_time
                pbar.set_description(f"AI Evaluation Progress | Last request: {duration:.1f}s")
                pbar.update(1)

            except Exception as e:
                results.append({
                    "dwa_id": dwa_id,
                    "dwa_title": dwa_title,
                    "score": np.nan,
                    "exposure": "ERROR",
                    "reason": f"Evaluation timeout: {str(e)}"
                })
                pbar.update(1)

    pbar.close()
    return pd.DataFrame(results)


def evaluate_dwa_batch(data: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
    """
    Evaluate a batch of DWA data, with optional sampling.

    Args:
        data (pd.DataFrame): DataFrame containing DWA data
        sample_size (int, optional): Number of samples to evaluate. If None, evaluates all.

    Returns:
        pd.DataFrame: Evaluation results
    """
    if data.empty:
        print("Error: No DWA data to evaluate")
        return pd.DataFrame()

    # Handle sampling if requested
    if sample_size and sample_size < len(data):
        print(f"Evaluating {sample_size} sample records")
        data = data.sample(sample_size, random_state=42).copy()
    else:
        print(f"Evaluating all {len(data)} DWA records")

    return evaluate_dwa_concurrent(data)


def save_results(results: pd.DataFrame, output_file: str = "dwa_exposure_scores.csv") -> pd.DataFrame:
    """
    Save evaluation results to a CSV file.

    Args:
        results (pd.DataFrame): DataFrame containing evaluation results
        output_file (str): Path to the output CSV file

    Returns:
        pd.DataFrame: The saved results with formatted columns
    """
    if results.empty:
        print("Error: No results to save")
        return pd.DataFrame()

    # Required columns check
    required_cols = ["dwa_id", "dwa_title", "score", "exposure", "reason"]
    missing_cols = [col for col in required_cols if col not in results.columns]

    if missing_cols:
        print(f"Error: Missing required fields in results: {missing_cols}")
        return pd.DataFrame()

    # Create bilingual column names
    col_map = {
        "dwa_id": "DWA_ID",
        "dwa_title": "DWA_Title",
        "score": "AI_Exposure_Score",
        "exposure": "Risk_Level",
        "reason": "Assessment_Reason"
    }

    # Rename columns and reorder
    results = results.rename(columns=col_map)
    ordered_cols = [col_map[col] for col in required_cols]
    results = results[ordered_cols]

    # Save to CSV
    results.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Results saved to: {output_file}")
    print(f"Total records: {len(results)}")

    # Display risk level distribution
    exposure_counts = results[col_map["exposure"]].value_counts()
    print("\nRisk Level Distribution:")
    for level, count in exposure_counts.items():
        print(f"{level}: {count} records")

    # Show preview of results
    preview = results.head(10).copy()
    print("\nResults Preview:")
    print(preview.to_string(index=False))

    return results


def main():
    """Main function to run the DWA AI exposure assessment tool."""
    print("DWA-AI Exposure Assessment Tool v2.0 (MetaChat Gemini Version)")
    print(f"Using model: {MODEL_ID}")
    print("Using MetaChat Gemini API")
    print("Includes exponential backoff retry mechanism for rate limiting\n")

    # Load DWA data
    file_path = "DWA_Reference.csv"
    print(f"Loading DWA data from: {file_path}")
    dwa_data = load_dwa_data(file_path)

    if dwa_data.empty:
        print("Error: Failed to load valid DWA data")
        exit(1)

    # Preview DWA titles
    print("\nFirst 5 DWA titles preview:")
    preview = dwa_data[["dwa_id", "dwa_title"]].head().copy()
    print(preview.to_string(index=False))

    # Sample evaluation
    try:
        sample_size = int(input("\nEnter sample size for test (0 = skip test): ") or 0)
        if sample_size > 0:
            sample_size = min(sample_size, len(dwa_data))
            print(f"\nStarting test evaluation ({sample_size} samples)...")
            test_results = evaluate_dwa_batch(dwa_data, sample_size)
            save_results(test_results, "dwa_exposure_scores_sample.csv")
    except ValueError:
        print("Invalid input, skipping test evaluation")

    # Full evaluation
    run_full = input("\nRun full evaluation? (y/n): ").lower() == 'y'
    if run_full:
        print(f"\nStarting full evaluation of {len(dwa_data)} DWA records...")
        final_results = evaluate_dwa_batch(dwa_data)
        save_results(final_results, "dwa_exposure_scores_full_gemini.csv")
        print("\nEvaluation completed!")
    else:
        print("Program ended")


if __name__ == "__main__":
    main()
