import os
import openai
from tqdm import tqdm

def read_file(filepath):
    """Read the content of a text file."""
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

def write_scores(output_dir, filename, scores):
    """Write the scores to a file in the output directory."""
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(','.join(map(str, scores)))

def evaluate_text(api_key, prompt, max_retries=3):
    """Send a prompt to the GPT API and ensure it returns a score."""
    openai.api_key = api_key
    for _ in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
            )
            content = response['choices'][0]['message']['content'].strip()
            if content.isdigit() and 1 <= int(content) <= 5:
                return int(content)
            elif content.replace('.', '', 1).isdigit() and 1 <= float(content) <= 5:
                return float(content)
        except Exception as e:
            print(f"Error during API call: {e}")
    raise ValueError("GPT did not return a valid score after multiple retries.")

def main(ground_truth_dir, evaluation_dir, output_dir, api_key):
    """Main function to evaluate texts and save scores."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gt_files = set(os.listdir(ground_truth_dir))
    eval_files = os.listdir(evaluation_dir)

    prompts = [
        """
        Evaluate the Completeness of Clue Extraction (CE):
        Definition: The completeness of clue extraction measures a model's ability to comprehensively cover all clues relevant to inference when processing text. The model should ensure that the extracted clues are complete to support sufficient and accurate inference, avoiding biases or errors due to missing important clues.
        Scoring Criteria:
        1. High (5): Clue extraction is complete, covering all cues in the GT, with comprehensive descriptions and no missing details. Examples: trees, lawns, terrain and soil clues, architectural styles (roofs, walls, solar panels), red mailboxes, and fence designs.
        2. Middle (2.5): Some clues are extracted, but not comprehensively. Examples: missing details like solar panels, red-brown soil, or temperate climate.
        3. Low (1): Clue extraction is severely lacking, missing many cues, vague descriptions, or low precision.
        Only provide a numerical score between 1 and 5. If the score is invalid, reevaluate.
        """,
        """
        Evaluate the Accuracy of Clue Extraction (AE):
        Definition: Accuracy of clue extraction assesses whether the model can accurately identify and extract key clues relevant for inference, excluding irrelevant or distracting clues.
        Scoring Criteria:
        1. High (5): Extracted clues are highly accurate and match the GT, effectively supporting correct inference. Examples: correct terrain, soil color, architectural features, and mailbox details.
        2. Middle (2.5): Some correct clues are extracted, but with minor inaccuracies. Examples: incorrect soil or wall color, vague mailbox description.
        3. Low (1): Extracted clues are highly inaccurate, leading to significant deviations from the GT. Examples: misidentifying terrain as desert or architectural styles as completely different.
        Only provide a numerical score between 1 and 5. If the score is invalid, reevaluate.
        """,
        """
        Evaluate the Correctness of the Relationship Between Inference and Clues (AC):
        Definition: This evaluates whether the model can correctly link extracted clues to conclusions during reasoning, avoiding interference from irrelevant clues or unreasonable conclusions.
        Scoring Criteria:
        1. High (5): Inference is accurate, based on clues, and consistent with the GT's logic. Examples: logical progression from cues to conclusions, utilizing relevant knowledge.
        2. Middle (2.5): Inference is partially correct, with some vague or incorrect conclusions. Examples: suggesting multiple potential locations instead of one specific match.
        3. Low (1): Inference is completely mismatched, with no logical connection between clues and conclusions.
        Only provide a numerical score between 1 and 5. If the score is invalid, reevaluate.
        """,
        """
        Evaluate the Reasonableness of the Inferential Logic of Clues (LC):
        Definition: This measures the coherence and logical consistency of the reasoning chain, ensuring conclusions align with common sense and contextual understanding.
        Scoring Criteria:
        1. High (5): Reasoning is coherent, logical, and aligns with the GT without contradictions. Examples: logical deductions from terrain and climate to architectural style and cultural clues.
        2. Middle (2.5): Reasoning has minor contradictions or logical errors but maintains general coherence. Examples: vague connections or unnecessary alternative inferences.
        3. Low (1): Severe logical contradictions or unreasonable conclusions, failing to integrate all clues.
        Only provide a numerical score between 1 and 5. If the score is invalid, reevaluate.
        """
    ]

    for eval_file in tqdm(eval_files, total=len(eval_files), desc="Processing files"):
        if eval_file in gt_files:
            eval_content = read_file(os.path.join(evaluation_dir, eval_file))
            gt_content = read_file(os.path.join(ground_truth_dir, eval_file))

            scores = []
            for prompt in prompts:
                full_prompt = f"Ground Truth:\n{gt_content}\n\nEvaluation Text:\n{eval_content}\n\n{prompt}"
                score = evaluate_text(api_key, full_prompt)
                scores.append(score)
                
            # print(f"处理文件: {eval_file}, 评分: {scores}")
            write_scores(output_dir, eval_file, scores)

if __name__ == "__main__":
    # Define your directories here
    ground_truth_dir = "data/gt_description"
    evaluation_dir = "data/llamas"
    output_dir = "data/llama_eval"
    api_key = "KKK"
    main(ground_truth_dir, evaluation_dir, output_dir, api_key)
