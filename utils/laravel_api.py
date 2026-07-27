import requests
import logging

logger = logging.getLogger(__name__)

BASE_API_URL = "http://127.0.0.1:8000/api"


def _fmt(val) -> str:
    """Safely format a value that might be a string like '85000.00' into 85,000."""
    try:
        n = float(val)
        return f"{int(n):,}" if n == int(n) else f"{n:,.2f}"
    except (ValueError, TypeError):
        return str(val) if val is not None else "0"


class LaravelAPI:
    """Fetches live data from the Nikky Frozen Laravel POS REST API."""

    # ------------------------------------------------------------------ #
    #  PRODUCTS  –  /api/products
    #  Stock lives inside  product.stocks[].stock  (per-branch).
    # ------------------------------------------------------------------ #
    @classmethod
    def get_products(cls) -> str:
        try:
            resp = requests.get(f"{BASE_API_URL}/products", timeout=10)
            resp.raise_for_status()
            raw = resp.json()
            products = raw.get("data", raw) if isinstance(raw, dict) else raw

            if not products:
                return "Tidak ada produk di database."

            lines = []
            for p in products:
                name     = p.get("name", "?")
                sku      = p.get("sku", "-")
                price    = _fmt(p.get("price", 0))
                category = p.get("category", "N/A")
                expiry   = p.get("expiry", "-")

                # --- aggregate stock per branch ---
                stocks = p.get("stocks", [])
                if stocks:
                    stock_parts = []
                    total_stock = 0
                    for s in stocks:
                        qty = s.get("stock", 0) or 0
                        total_stock += qty
                        branch_name = "?"
                        branch = s.get("branch")
                        if isinstance(branch, dict):
                            branch_name = branch.get("name", "?")
                        stock_parts.append(f"{branch_name}: {qty}")
                    stock_info = f"{total_stock} unit total ({', '.join(stock_parts)})"
                else:
                    total_stock = 0
                    stock_info = "0 unit (belum ada data stok)"

                restock_warning = ""
                if total_stock <= 10:
                    restock_warning = " ⚠️ PERLU RESTOCK!"

                lines.append(
                    f"• {name} (SKU: {sku}) | Kategori: {category} | "
                    f"Harga: Rp{price} | Kedaluwarsa: {expiry} | "
                    f"Stok: {stock_info}{restock_warning}"
                )

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"get_products error: {e}")
            return f"[ERROR] Gagal mengambil data produk dari Laravel: {e}"

    # ------------------------------------------------------------------ #
    #  SALES  –  /api/sales
    #  Fields: invoice_number, total, payment_method, payment_status,
    #          items_qty, user.name, branch.name
    # ------------------------------------------------------------------ #
    @classmethod
    def get_sales(cls) -> str:
        try:
            resp = requests.get(f"{BASE_API_URL}/sales", timeout=10)
            resp.raise_for_status()
            raw = resp.json()
            sales = raw.get("data", raw) if isinstance(raw, dict) else raw

            if not sales:
                return "Belum ada transaksi penjualan."

            lines = []
            for s in sales:
                inv    = s.get("invoice_number", "-")
                total  = _fmt(s.get("total", 0))
                method = s.get("payment_method", "-")
                status = s.get("payment_status", "-")
                qty    = s.get("items_qty", 0)
                date   = s.get("created_at", "")[:10]

                cashier = "?"
                user = s.get("user")
                if isinstance(user, dict):
                    cashier = user.get("name", "?")

                branch_name = "?"
                branch = s.get("branch")
                if isinstance(branch, dict):
                    branch_name = branch.get("name", "?")

                lines.append(
                    f"• {inv} | Kasir: {cashier} | Cabang: {branch_name} | "
                    f"Jumlah Item: {qty} | Total: Rp{total} | "
                    f"Bayar: {method} | Status: {status} | Tanggal: {date}"
                )

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"get_sales error: {e}")
            return f"[ERROR] Gagal mengambil data penjualan dari Laravel: {e}"

    # ------------------------------------------------------------------ #
    #  DASHBOARD SUMMARY  –  /api/dashboard/summary
    # ------------------------------------------------------------------ #
    @classmethod
    def get_dashboard_summary(cls) -> str:
        try:
            resp = requests.get(f"{BASE_API_URL}/dashboard/summary", timeout=10)
            resp.raise_for_status()
            d = resp.json()

            total_products  = d.get("totalProducts", 0)
            total_stocks    = _fmt(d.get("totalStocks", 0))
            low_stock       = d.get("lowStockCount", 0)
            expiring_count  = d.get("expiringCount", 0)
            today_revenue   = _fmt(d.get("todayRevenue", 0))
            total_revenue   = _fmt(d.get("totalRevenue", 0))
            users_count     = d.get("usersCount", 0)
            total_transfers = d.get("totalTransfers", 0)

            # Branch revenues
            branches = d.get("branches", [])
            branch_lines = []
            for b in branches:
                bname = b.get("name", "?")
                brev  = _fmt(b.get("revenue", 0))
                branch_lines.append(f"  - {bname}: Rp{brev}")

            # Category distribution
            categories = d.get("categoryData", [])
            cat_lines = []
            for c in categories:
                cat_lines.append(f"  - {c.get('name', '?')}: {c.get('value', 0)}%")

            summary = (
                f"=== RINGKASAN BISNIS NIKKY FROZEN ===\n"
                f"Total Produk: {total_products}\n"
                f"Total Stok Seluruh Cabang: {total_stocks} unit\n"
                f"Produk Stok Rendah: {low_stock}\n"
                f"Produk Mendekati Kedaluwarsa: {expiring_count}\n"
                f"Revenue Hari Ini: Rp{today_revenue}\n"
                f"Total Revenue: Rp{total_revenue}\n"
                f"Total Transfer Stok: {total_transfers}\n"
                f"Jumlah User: {users_count}\n"
            )

            if branch_lines:
                summary += f"\nRevenue Per Cabang:\n" + "\n".join(branch_lines) + "\n"
            if cat_lines:
                summary += f"\nDistribusi Kategori:\n" + "\n".join(cat_lines) + "\n"

            return summary

        except Exception as e:
            logger.error(f"get_dashboard_summary error: {e}")
            return f"[ERROR] Gagal mengambil ringkasan dashboard: {e}"
