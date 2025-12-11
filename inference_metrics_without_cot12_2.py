import csv

CSV_PATH = "/direct_predictions2.csv"

def compute_metrics(csv_file):

    total = 0
    correct = 0

    yes_gt = 0
    yes_correct = 0

    no_gt = 0
    no_correct = 0

    numeric_gt = 0
    numeric_correct = 0

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            gt = row["gt_final_answer"].strip().lower()
            pred = row["predicted_final_answer"].strip().lower()

            # Only count samples with a prediction
            if pred == "":
                continue

            total += 1

            # overall accuracy
            if gt == pred:
                correct += 1

            # yes accuracy
            if gt == "yes":
                yes_gt += 1
                if pred == "yes":
                    yes_correct += 1

            # no accuracy
            elif gt == "no":
                no_gt += 1
                if pred == "no":
                    no_correct += 1

            # numeric grade accuracy
            else:
                # gt is something like 0/1/2/3
                if gt.isdigit():
                    numeric_gt += 1
                    if pred == gt:
                        numeric_correct += 1

    # Avoid division by zero
    overall_acc = correct / total if total > 0 else 0
    yes_acc = yes_correct / yes_gt if yes_gt > 0 else 0
    no_acc = no_correct / no_gt if no_gt > 0 else 0
    numeric_acc = numeric_correct / numeric_gt if numeric_gt > 0 else 0

    print("\n========== METRICS ==========")
    print(f"Total samples:            {total}")
    print(f"Overall accuracy:         {overall_acc:.4f}")
    print(f"YES accuracy:             {yes_acc:.4f}   (GT yes: {yes_gt})")
    print(f"NO accuracy:              {no_acc:.4f}    (GT no:  {no_gt})")
    print(f"Numeric grade accuracy:   {numeric_acc:.4f} (GT numeric: {numeric_gt})")
    print("================================\n")


if __name__ == "__main__":
    compute_metrics(CSV_PATH)
