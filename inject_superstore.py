import pandas as pd
import datetime
import random

print("Reading superstore.csv...")
df = pd.read_csv("datasets/superstore.csv", encoding='latin1')

print("Generating SQL...")
with open("import_superstore.sql", "w", encoding='utf-8') as f:
    f.write("SET FOREIGN_KEY_CHECKS = 0;\n")
    f.write("TRUNCATE TABLE sale_items;\n")
    f.write("TRUNCATE TABLE sales;\n")
    f.write("TRUNCATE TABLE products;\n")
    f.write("TRUNCATE TABLE categories;\n\n")

    # Categories
    categories = df['Category'].dropna().unique()
    if len(categories) > 0:
        f.write("INSERT INTO categories (name, created_at, updated_at) VALUES\n")
        cat_vals = []
        for c in categories:
            safe_c = str(c).replace("'", "''")
            cat_vals.append(f"('{safe_c}', NOW(), NOW())")
        f.write(",\n".join(cat_vals) + ";\n\n")

    # Products
    df['Price'] = df['Sales'] / df['Quantity']
    product_df = df.groupby('Product.ID').agg({
        'Product.Name': 'first',
        'Category': 'first',
        'Price': 'mean'
    }).reset_index()
    
    product_mapping = {} 
    
    if len(product_df) > 0:
        prod_vals = []
        for i, row in product_df.iterrows():
            db_id = i + 1
            product_mapping[row['Product.ID']] = db_id
            
            safe_name = str(row['Product.Name']).replace("'", "''")
            safe_cat = str(row['Category']).replace("'", "''")
            sku = str(row['Product.ID']).replace("'", "''")
            price = round(row['Price'], 2)
            
            prod_vals.append(f"({db_id}, '{sku}', '{safe_name}', '{safe_cat}', {price}, '2030-01-01', NULL, NOW(), NOW(), 1)")
        
        batch_size = 1000
        for i in range(0, len(prod_vals), batch_size):
            batch = prod_vals[i:i+batch_size]
            f.write("INSERT INTO products (id, sku, name, category, price, expiry, image, created_at, updated_at, branch_id) VALUES\n")
            f.write(",\n".join(batch) + ";\n")
        f.write("\n")

    # Sales & Sale Items
    sales_df = df.groupby('Order.ID').agg({
        'Order.Date': 'first',
        'Sales': 'sum'
    }).reset_index()
    sale_mapping = {}
    
    if len(sales_df) > 0:
        sale_vals = []
        for i, row in sales_df.iterrows():
            db_id = i + 1
            sale_mapping[row['Order.ID']] = db_id
            
            inv = str(row['Order.ID']).replace("'", "''")
            total = round(row['Sales'], 2)
            raw_date = str(row['Order.Date'])
            try:
                dt = pd.to_datetime(raw_date).strftime('%Y-%m-%d %H:%M:%S')
            except:
                dt = '2024-01-01 12:00:00'
                
            sale_vals.append(f"({db_id}, '{inv}', 1, {total}, 'cash', 'paid', '{dt}', '{dt}', 1)")
            
        batch_size = 1000
        for i in range(0, len(sale_vals), batch_size):
            batch = sale_vals[i:i+batch_size]
            f.write("INSERT INTO sales (id, invoice_number, user_id, total, payment_method, payment_status, created_at, updated_at, branch_id) VALUES\n")
            f.write(",\n".join(batch) + ";\n")
        f.write("\n")

    # Sale Items
    sale_items_vals = []
    si_id = 1
    for i, row in df.iterrows():
        order_id = row['Order.ID']
        prod_id = row['Product.ID']
        if order_id in sale_mapping and prod_id in product_mapping:
            s_db_id = sale_mapping[order_id]
            p_db_id = product_mapping[prod_id]
            qty = row['Quantity']
            subtotal = round(row['Sales'], 2)
            price = round(subtotal / qty, 2)
            
            sale_items_vals.append(f"({si_id}, {s_db_id}, {p_db_id}, {qty}, {price}, {subtotal}, NOW(), NOW())")
            si_id += 1

    batch_size = 1000
    for i in range(0, len(sale_items_vals), batch_size):
        batch = sale_items_vals[i:i+batch_size]
        f.write("INSERT INTO sale_items (id, sale_id, product_id, qty, price, subtotal, created_at, updated_at) VALUES\n")
        f.write(",\n".join(batch) + ";\n")

    f.write("\nSET FOREIGN_KEY_CHECKS = 1;\n")

print("Generated import_superstore.sql successfully!")
