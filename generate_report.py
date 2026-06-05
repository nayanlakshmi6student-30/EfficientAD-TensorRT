import json
import numpy as np

def calculate_metrics(json_path, category_name, total_normal, total_anomalous):
    """Parses JSON data to calculate optimal threshold metrics."""
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    cat_data = data[category_name]
    fpr = np.array(cat_data["classification_roc_curve_fpr"])
    tpr = np.array(cat_data["classification_roc_curve_tpr"])
    auroc = cat_data["classification_au_roc"]
    au_pro = cat_data.get("au_pro", 0.0)
    
    # Locate the optimal index using Youden's Index (Maximizing TPR - FPR)
    youden_index = np.argmax(tpr - fpr)
    best_fpr = fpr[youden_index]
    best_tpr = tpr[youden_index]
    
    # Derive Confusion Matrix elements based on actual MVTec test dataset splits
    fp = int(round(best_fpr * total_normal))
    tn = total_normal - fp
    tp = int(round(best_tpr * total_anomalous))
    fn = total_anomalous - tp
    
    # Calculate target validation metrics
    accuracy = (tp + tn) / (total_normal + total_anomalous)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = best_tpr
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "auroc": auroc,
        "au_pro": au_pro,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# File names (Make sure these JSONs are in the same folder on your remote machine!)
models = {
    "Small Model Variant": "mvtec_ad_small.json",
    "Medium Model Variant": "mvtec_ad_medium.json"
}

# Standard test set distributions for MVTec AD dataset
categories = {
    "bottle": {"normal": 20, "anomalous": 63},
    "cable": {"normal": 58, "anomalous": 92}
}

output_filename = "evaluation_report.txt"

# Generate and write report to file
with open(output_filename, "w") as out:
    out.write("=========================================================\n")
    out.write("     DHVANI INTERNSHIP: ANOMALY DETECTION EVALUATION     \n")
    out.write("=========================================================\n\n")
    
    for model_name, json_file in models.items():
        out.write(f"MODEL CONFIGURATION: {model_name} ({json_file})\n")
        out.write("=" * 60 + "\n")
        
        for cat, counts in categories.items():
            try:
                res = calculate_metrics(json_file, cat, counts["normal"], counts["anomalous"])
                
                out.write(f"Category: {cat.upper()}\n")
                out.write(f"  - Classification AUROC : {res['auroc']*100:.2f}%\n")
                out.write(f"  - Localization AU-PRO  : {res['au_pro']*100:.2f}%\n")
                out.write(f"  - Accuracy             : {res['accuracy']*100:.2f}%\n")
                out.write(f"  - Precision            : {res['precision']*100:.2f}%\n")
                out.write(f"  - F1-Score             : {res['f1']*100:.2f}%\n")
                out.write("\n")
                out.write("  Confusion Matrix:\n")
                out.write(f"    [ Predicted NORMAL ] -> TN: {res['tn']:3d} | FP (Overkill): {res['fp']:3d}\n")
                out.write(f"    [ Predicted DEFECT ] -> FN: {res['fn']:3d} | TP (Detected): {res['tp']:3d}\n")
                out.write(f"    (Total Ground Truth -> Normal: {counts['normal']}, Anomalous: {counts['anomalous']})\n")
                out.write("-" * 40 + "\n\n")
            except FileNotFoundError:
                out.write(f"ERROR: Could not find file '{json_file}' in this directory.\n\n")
            except KeyError:
                out.write(f"ERROR: Category '{cat}' not found inside '{json_file}'.\n\n")
                
        out.write("\n" + "#" * 60 + "\n\n")

print(f"Success! The comprehensive report has been saved to: {output_filename}")