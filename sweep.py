import pandas as pd
import json
from transformers import AutoTokenizer, set_seed
from lab import prepare_dataset, tokenize_dataset, make_training_args, train_classifier, evaluate_classifier, get_data_path

def run_hyperparameter_sweep():
    # 1. إعدادات أساسية
    set_seed(42)
    data_path = get_data_path()
    model_name = "distilbert-base-uncased"
    
    # القيم اللي بدنا نجربها (Tier 1)
    learning_rates = [1e-5, 5e-5, 1e-4]
    sweep_results = []

    print("🚀 Starting Hyperparameter Sweep...")

    # 2. تجهيز البيانات مرة واحدة لتوفير الوقت
    ds = prepare_dataset(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenize_dataset(ds, tokenizer)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # 3. دوران على كل قيمة Learning Rate
    for lr in learning_rates:
        print(f"\n--- Testing Learning Rate: {lr} ---")
        
        # إنشاء مجلد منفصل لكل تجربة
        output_dir = f"model_lr_{lr}"
        
        # استخدام دالتك من lab.py
        args = make_training_args(output_dir, lr=lr)
        
        # التدريب والتقييم
        trainer = train_classifier(tokenized, model_name, args, tokenizer)
        metrics = evaluate_classifier(trainer, tokenized["test"])
        
        # تخزين النتيجة
        sweep_results.append({
            "learning_rate": lr,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"]
        })

    # 4. عرض وحفظ النتائج
    df = pd.DataFrame(sweep_results)
    print("\n✅ Sweep Finished! Final Results:")
    print(df.to_string(index=False))
    
    df.to_csv("sweep_results.csv", index=False)
    print("\nResults saved to 'sweep_results.csv'")

if __name__ == "__main__":
    run_hyperparameter_sweep()