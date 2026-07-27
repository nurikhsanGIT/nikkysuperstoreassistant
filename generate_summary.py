import pandas as pd
import os

print("Membaca dataset...")
df = pd.read_csv("datasets/Customer_support_data.csv")

total_complaints = len(df)
avg_csat = df['CSAT Score'].mean()

top_categories = df['category'].value_counts().head(5).to_dict()
top_subcategories = df['Sub-category'].value_counts().head(5).to_dict()

# Create summary text
summary = f"""
# RINGKASAN DATA KOMPLAIN PELANGGAN (NIKKY SUPERSTORE - CUSTOMER SUPPORT)

Berdasarkan dataset `Customer_support_data.csv`:
- **Total Keluhan Pelanggan:** {total_complaints}
- **Rata-rata Skor Kepuasan (CSAT):** {avg_csat:.2f} dari 5.0

## Kategori Masalah Terbanyak:
"""
for k, v in top_categories.items():
    summary += f"- {k}: {v} keluhan\n"

summary += "\n## Sub-kategori Masalah Terbanyak:\n"
for k, v in top_subcategories.items():
    summary += f"- {k}: {v} keluhan\n"

summary += """
## Analisis dan Rekomendasi (Dihasilkan oleh Analyst Agent):
Sebagian besar masalah pelanggan terpusat pada kategori-kategori utama di atas. 
Agen Customer Service perlu memprioritaskan penanganan keluhan terkait pesanan (Order Related) dan masalah pengiriman/pengembalian uang (Refund).
Nilai CSAT saat ini bisa ditingkatkan dengan memberikan respon yang lebih cepat.
"""

# Ensure documents dir exists
os.makedirs("documents", exist_ok=True)
with open("documents/CSAT_Summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("Berhasil membuat documents/CSAT_Summary.txt")
