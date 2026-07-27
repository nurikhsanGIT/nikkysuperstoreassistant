import logging
import requests
from utils.laravel_api import LaravelAPI, BASE_API_URL

logger = logging.getLogger(__name__)

class LaravelTool:
    """Wrapper class for communicating with the Laravel POS API."""
    
    @staticmethod
    def get_products() -> str:
        return LaravelAPI.get_products()

    @staticmethod
    def get_sales() -> str:
        return LaravelAPI.get_sales()

    @staticmethod
    def get_dashboard_summary() -> str:
        return LaravelAPI.get_dashboard_summary()
