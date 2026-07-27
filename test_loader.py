import sys
sys.stdout.reconfigure(encoding='utf-8')

from utils.data_loader import SuperstoreDataLoader

# Test 1: load superstore
df = SuperstoreDataLoader.get_superstore()
cols = list(df.columns[:6])
print(f"[OK] Superstore loaded: {len(df)} rows")
print(f"     Columns sample: {cols}")

# Test 2: top products
top = SuperstoreDataLoader.get_top_products(top_n=3)
print(f"\n[OK] Top 3 Products:")
for _, r in top.iterrows():
    print(f"     - {r['Product_Name']} | {r['Category']} | Sales: ${r['Total_Sales']:,.2f}")

# Test 3: finance summary
fin = SuperstoreDataLoader.get_finance_summary()
print(f"\n[OK] Finance Summary: {fin}")

# Test 4: customer support CSAT
cs = SuperstoreDataLoader.get_csat_summary()
print(f"\n[OK] CSAT Summary: {cs}")

# Test 5: products by category
inv = SuperstoreDataLoader.get_products(category="Technology", top_n=3)
print(f"\n[OK] Technology products (top 3):")
for _, r in inv.iterrows():
    print(f"     - {r['Product_Name']} | Sales: ${r['Total_Sales']:,.2f}")

print("\n=== All tests passed! ===")
