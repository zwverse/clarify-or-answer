import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(file_path):
    y_true = []
    y_pred = []
    invalid_outputs = 0

    with open(file_path, 'r') as f:
        for line in f:
            try:
                line = line.strip()
                if not line: continue
                if not line.startswith('{'):
                    line = line[line.find('{'):]

                data = json.loads(line)

                # Normalize values to lowercase and strip whitespace
                true_val = str(data['labels']).lower().strip()
                pred_val = str(data['response']).lower().strip()

                # Robust mapping
                label_map = {'yes': 1, 'no': 0}

                if true_val in label_map and pred_val in label_map:
                    y_true.append(label_map[true_val])
                    y_pred.append(label_map[pred_val])
                else:
                    # If pred is 'east' or other text, count it as 'no' (0)
                    # because it provided an answer instead of asking for clarification
                    # y_true.append(label_map.get(true_val, 0))
                    # y_pred.append(0)
                    invalid_outputs += 1

            except Exception as e:
                print(f"Skipping malformed line: {e}")

    if not y_true:
        print("No valid data found.")
        return

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Handle Confusion Matrix even if some classes are missing
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"--- Results for {file_path} ---")
    print(f"Total Samples: {len(y_true)}")
    print(f"Invalid/Specific Outputs (omit): {invalid_outputs}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\n--- Confusion Matrix ---")
    print(f"TP: {tp}, FP: {fp}")
    print(f"TN: {tn}, FN: {fn}")

if __name__ == "__main__":
    # Point this to your actual file path
    calculate_metrics(
        '/home/clarify-ambiguity/output/internvl3_2b_router_sft/v0-20251228-094231/checkpoint-46/infer_result/20260105-075004.jsonl')