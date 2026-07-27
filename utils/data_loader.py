import os
import logging
import pandas as pd
from functools import lru_cache
from utils.config import BASE_DIR

logger = logging.getLogger(__name__)

SUPERSTORE_PATH = os.path.join(BASE_DIR, "datasets", "superstore.csv")
CUSTOMER_SUPPORT_PATH = os.path.join(BASE_DIR, "datasets", "Customer_support_data.csv")


class SuperstoreDataLoader:
    """
    Singleton loader untuk dataset superstore.csv dan Customer_support_data.csv.
    Semua agent harus pakai class ini — bukan MySQL.
    """

    _superstore_df: pd.DataFrame = None
    _customer_support_df: pd.DataFrame = None

    @classmethod
    def get_superstore(cls) -> pd.DataFrame:
        if cls._superstore_df is None:
            try:
                cls._superstore_df = pd.read_csv(
                    SUPERSTORE_PATH,
                    encoding="latin-1",
                    low_memory=False
                )
                # Rename kolom agar lebih bersih
                cls._superstore_df.columns = [
                    c.strip().replace(".", "_") for c in cls._superstore_df.columns
                ]
                # Pastikan tipe numerik
                for col in ["Sales", "Profit", "Quantity", "Discount", "Shipping_Cost"]:
                    if col in cls._superstore_df.columns:
                        cls._superstore_df[col] = pd.to_numeric(
                            cls._superstore_df[col], errors="coerce"
                        ).fillna(0)
                logger.info(f"Superstore dataset loaded: {len(cls._superstore_df)} rows.")
            except Exception as e:
                logger.error(f"Failed to load superstore.csv: {e}")
                cls._superstore_df = pd.DataFrame()
        return cls._superstore_df

    @classmethod
    def get_customer_support(cls) -> pd.DataFrame:
        if cls._customer_support_df is None:
            try:
                cls._customer_support_df = pd.read_csv(
                    CUSTOMER_SUPPORT_PATH,
                    encoding="utf-8",
                    low_memory=False
                )
                logger.info(f"Customer support dataset loaded: {len(cls._customer_support_df)} rows.")
            except Exception as e:
                logger.error(f"Failed to load Customer_support_data.csv: {e}")
                cls._customer_support_df = pd.DataFrame()
        return cls._customer_support_df

    # ── Superstore helpers ────────────────────────────────────────────────────

    @classmethod
    def get_products(cls, keyword: str = None, category: str = None,
                     sub_category: str = None, top_n: int = 20) -> pd.DataFrame:
        """Kembalikan daftar produk unik dengan total sales & profit."""
        df = cls.get_superstore()
        if df.empty:
            return df

        mask = pd.Series([True] * len(df))
        if keyword:
            kw = keyword.lower()
            mask &= df["Product_Name"].str.lower().str.contains(kw, na=False)
        if category:
            mask &= df["Category"].str.lower().str.contains(category.lower(), na=False)
        if sub_category:
            mask &= df["Sub_Category"].str.lower().str.contains(sub_category.lower(), na=False)

        filtered = df[mask]
        result = (
            filtered.groupby(["Product_Name", "Category", "Sub_Category"], as_index=False)
            .agg(
                Total_Sales=("Sales", "sum"),
                Total_Profit=("Profit", "sum"),
                Total_Quantity=("Quantity", "sum"),
            )
            .sort_values("Total_Sales", ascending=False)
            .head(top_n)
        )
        result["Total_Sales"] = result["Total_Sales"].round(2)
        result["Total_Profit"] = result["Total_Profit"].round(2)
        return result

    @classmethod
    def get_top_products(cls, top_n: int = 10) -> pd.DataFrame:
        """Top N produk berdasarkan total penjualan."""
        df = cls.get_superstore()
        if df.empty:
            return df
        result = (
            df.groupby(["Product_Name", "Category", "Sub_Category"], as_index=False)
            .agg(Total_Sales=("Sales", "sum"), Total_Quantity=("Quantity", "sum"),
                 Total_Profit=("Profit", "sum"))
            .sort_values("Total_Sales", ascending=False)
            .head(top_n)
        )
        result["Total_Sales"] = result["Total_Sales"].round(2)
        result["Total_Profit"] = result["Total_Profit"].round(2)
        return result

    @classmethod
    def get_slow_products(cls, bottom_n: int = 10) -> pd.DataFrame:
        """Bottom N produk (slow moving) berdasarkan total quantity."""
        df = cls.get_superstore()
        if df.empty:
            return df
        result = (
            df.groupby(["Product_Name", "Category", "Sub_Category"], as_index=False)
            .agg(Total_Quantity=("Quantity", "sum"), Total_Sales=("Sales", "sum"),
                 Total_Profit=("Profit", "sum"))
            .sort_values("Total_Quantity", ascending=True)
            .head(bottom_n)
        )
        result["Total_Sales"] = result["Total_Sales"].round(2)
        result["Total_Profit"] = result["Total_Profit"].round(2)
        return result

    @classmethod
    def get_finance_summary(cls) -> dict:
        """Ringkasan keuangan dari seluruh dataset."""
        df = cls.get_superstore()
        if df.empty:
            return {}
        return {
            "total_sales": round(float(df["Sales"].sum()), 2),
            "total_profit": round(float(df["Profit"].sum()), 2),
            "total_quantity": int(df["Quantity"].sum()),
            "avg_discount": round(float(df["Discount"].mean()) * 100, 2),
            "total_orders": int(df["Order_ID"].nunique()) if "Order_ID" in df.columns else 0,
            "total_customers": int(df["Customer_ID"].nunique()) if "Customer_ID" in df.columns else 0,
        }

    @classmethod
    def get_finance_by_category(cls) -> pd.DataFrame:
        """Sales & profit per kategori."""
        df = cls.get_superstore()
        if df.empty:
            return df
        result = (
            df.groupby("Category", as_index=False)
            .agg(Total_Sales=("Sales", "sum"), Total_Profit=("Profit", "sum"),
                 Total_Quantity=("Quantity", "sum"))
            .sort_values("Total_Sales", ascending=False)
        )
        result["Total_Sales"] = result["Total_Sales"].round(2)
        result["Total_Profit"] = result["Total_Profit"].round(2)
        return result

    @classmethod
    def get_sales_by_region(cls) -> pd.DataFrame:
        """Penjualan per region."""
        df = cls.get_superstore()
        if df.empty:
            return df
        result = (
            df.groupby("Region", as_index=False)
            .agg(Total_Sales=("Sales", "sum"), Total_Profit=("Profit", "sum"),
                 Total_Orders=("Order_ID", "nunique"))
            .sort_values("Total_Sales", ascending=False)
        )
        result["Total_Sales"] = result["Total_Sales"].round(2)
        result["Total_Profit"] = result["Total_Profit"].round(2)
        return result

    @classmethod
    def get_sales_by_segment(cls) -> pd.DataFrame:
        """Penjualan per segmen pelanggan."""
        df = cls.get_superstore()
        if df.empty:
            return df
        result = (
            df.groupby("Segment", as_index=False)
            .agg(Total_Sales=("Sales", "sum"), Total_Profit=("Profit", "sum"))
            .sort_values("Total_Sales", ascending=False)
        )
        result["Total_Sales"] = result["Total_Sales"].round(2)
        result["Total_Profit"] = result["Total_Profit"].round(2)
        return result

    # ── Customer Support helpers ──────────────────────────────────────────────

    @classmethod
    def get_csat_summary(cls) -> dict:
        """Ringkasan CSAT dari dataset Customer Support."""
        df = cls.get_customer_support()
        if df.empty:
            return {}
        csat_col = "CSAT Score"
        if csat_col not in df.columns:
            return {}
        return {
            "avg_csat": round(float(df[csat_col].mean()), 2),
            "total_complaints": len(df),
            "top_category": str(df["category"].mode()[0]) if "category" in df.columns else "N/A",
            "csat_distribution": df[csat_col].value_counts().to_dict(),
        }

    @classmethod
    def get_complaints_by_category(cls, top_n: int = 5) -> pd.DataFrame:
        """Kategori komplain terbanyak."""
        df = cls.get_customer_support()
        if df.empty or "category" not in df.columns:
            return pd.DataFrame()
        return (
            df.groupby("category", as_index=False)
            .agg(Total=("category", "count"), Avg_CSAT=("CSAT Score", "mean"))
            .sort_values("Total", ascending=False)
            .head(top_n)
        )
