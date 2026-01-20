import json
import os

from rewards import (
    question_format_reward,
    question_focused_relevance_reward,
    novelty_reward,
    ground_truth_similarity_reward,
    ambiguity_resolution_reward,
)

def evaluate_from_jsonl(file_path):
    data_list = []

    # 1. Load the JSONL file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_list.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    # 2. Prepare inputs for reward functions
    base_image_dir = "/clarify-ambiguity/GRIT_data/clearvqa/images"

    prompts = [d.get("question", "") for d in data_list]
    completions = [d.get("sft_output", "") for d in data_list]
    gt_answers = [d.get("clarification_request", "") for d in data_list]
    category = [d.get("category", "") for d in data_list]
    dataset = [d.get("dataset", "") for d in data_list]
    images = [os.path.join(base_image_dir, d.get("image", "")) for d in data_list]    # Passing the original question as the 'question' kwarg
    kwargs_data = {"question": prompts, "category": category, "dataset": dataset, "image": images}

    # 3. Calculate all rewards
    # We pass the same lists to all functions as per your requirement
    fmt_rewards = question_format_reward("", completions, "", **kwargs_data)
    rel_rewards = question_focused_relevance_reward("", completions, "", **kwargs_data)
    nov_rewards = novelty_reward("", completions, "", **kwargs_data)
    sim_rewards = ground_truth_similarity_reward("", completions, gt_answers, **kwargs_data)
    amb_rewards = ambiguity_resolution_reward("", completions, "", **kwargs_data)

    # 4. Output Detailed Table
    header = f"{'ID':<5} | {'Format':<6} | {'Rel':<5} | {'Nov':<5} | {'Sim':<5} | {'Amb':<5} | {'Total'}"
    print(header)
    print("-" * len(header))

    for i in range(len(data_list)):
        item_id = data_list[i].get("id", "N/A")

        # Calculate a simple sum for the total score
        total = fmt_rewards[i] + rel_rewards[i] + nov_rewards[i] + sim_rewards[i] + amb_rewards[i]

        print(f"{item_id:<5} | "
              f"{fmt_rewards[i]:>6.2f} | "
              f"{rel_rewards[i]:>5.2f} | "
              f"{nov_rewards[i]:>5.2f} | "
              f"{sim_rewards[i]:>5.2f} | "
              f"{amb_rewards[i]:>5.2f} | "
              f"{total:>6.2f}")

    # 5. Summary Statistics
    print("-" * len(header))
    metrics = {
        "Format": fmt_rewards,
        "Relevance": rel_rewards,
        "Novelty": nov_rewards,
        "Similarity": sim_rewards,
        "Ambiguity": amb_rewards
    }

    print("Average Scores:")
    for name, values in metrics.items():
        avg = sum(values) / len(values) if values else 0
        print(f"- {name:<10}: {avg:.3f}")

if __name__ == "__main__":
    evaluate_from_jsonl("sft_test_result.jsonl")