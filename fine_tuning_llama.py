import pandas as pd
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset

# ==========================================
# 1. PERSIAPAN DATASET NIKKY FROZEN
# ==========================================
print("Memuat dataset Customer Support...")
df = pd.read_csv("datasets/Customer_support_data.csv")

# Kita asumsikan dataset ini adalah keluhan pelanggan dari cabang-cabang Nikky Frozen.
# Karena dataset aslinya sangat besar (85.000+ baris), untuk Fine-Tuning kita 
# hanya akan mengambil subset data yang valid (misalnya memiliki Remarks dan CSAT).
df_clean = df.dropna(subset=["Customer Remarks", "CSAT Score"]).copy()

# Buat klasifikasi sentimen berdasarkan CSAT Score
# CSAT 4-5 = Positif, CSAT 3 = Netral, CSAT 1-2 = Negatif
def get_sentiment(csat):
    if csat >= 4: return "Positif"
    elif csat == 3: return "Netral"
    else: return "Negatif"

df_clean["Sentiment"] = df_clean["CSAT Score"].apply(get_sentiment)

# Ambil 1000 sampel untuk mempercepat proses training (demonstrasi)
df_sample = df_clean.sample(1000, random_state=42)

# Format dataset menjadi Prompt Instruction yang dimengerti Llama
def format_prompt(row):
    instruction = f"Analisislah keluhan pelanggan Nikky Frozen POS berikut. Kategori: {row['category']}. Sub-kategori: {row['Sub-category']}."
    input_text = row['Customer Remarks']
    response = f"Berdasarkan ulasan, CSAT pelanggan ini adalah {row['CSAT Score']}/5 ({row['Sentiment']}). Tindak lanjut disarankan untuk memperbaiki {row['category']}."
    
    # Llama 3 Prompt Format
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nAnda adalah agen Customer Service Analyst di Nikky Frozen.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\nKeluhan: {input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
    return prompt

df_sample["text"] = df_sample.apply(format_prompt, axis=1)

# Ubah pandas dataframe menjadi HuggingFace Dataset
hg_dataset = Dataset.from_pandas(df_sample[["text"]])
print("Format dataset selesai. Contoh Prompt:")
print(hg_dataset["text"][0])


# ==========================================
# 2. KONFIGURASI MODEL & LoRA (PEFT)
# ==========================================
# Catatan: Bagian ini membutuhkan GPU (VRAM >= 8GB). 
# Jika dijalankan tanpa GPU, proses ini hanya simulasi.

model_id = "meta-llama/Meta-Llama-3-8B-Instruct" # Model dasar Llama 3

print("\nMengonfigurasi LoRA Adapter...")
# Parameter LoRA (Low-Rank Adaptation)
# Ini adalah teknik Fine-Tuning yang efisien karena hanya melatih sebagian kecil parameter model.
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

# Dalam eksekusi nyata, kode di bawah ini akan di-uncomment untuk meload model ke VRAM
'''
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True # Kuantisasi agar muat di memori kecil
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
'''

# ==========================================
# 3. PROSES FINE-TUNING (SFTTrainer)
# ==========================================
print("\nMenyiapkan Parameter Training (Epochs, Batch Size, dll)...")
training_args = TrainingArguments(
    output_dir="./llama3-nikky-frozen-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=100, # Batasi step untuk percobaan
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

# Simulasi SFTTrainer
'''
trainer = SFTTrainer(
    model=model,
    train_dataset=hg_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

print("\nMemulai Proses Fine-Tuning Llama 3...")
trainer.train()

# Simpan model yang sudah di-Fine Tune
trainer.model.save_pretrained("llama3-nikky-frozen-final")
tokenizer.save_pretrained("llama3-nikky-frozen-final")
print("Proses Fine-Tuning Selesai!")
'''

print("\n[SIMULASI BERHASIL] - Kode Fine-Tuning Llama 3 dengan dataset Customer Support untuk sistem Nikky Frozen POS telah siap dipresentasikan.")
