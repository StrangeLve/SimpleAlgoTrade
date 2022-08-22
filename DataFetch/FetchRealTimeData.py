import requests
from joblib import Parallel, delayed
from typing import Tuple, Dict, List, Union

class FetchRealTimeData:
    API_URL = "https://api.binance.com"
    TEST_API ="/api/v3/ping"
    PRICE_URL = "/api/v3/ticker/price?symbol={symbol}"
    ORDER_BOOK = "/api/v3/depth?symbol={symbol}&limit={limit}"
    ROLLING_STATS = "/api/v3/ticker?symbol={symbol}&windowSize=1m"

    def __init__(self,
                 symbol: str="BTCUSDT",
                 trade_book_limit: int = 100):
        self.symbol = symbol
        self.trade_book_limit = trade_book_limit
        assert self.test_connection() == {}, "Connection Failed"

    def get_data(self,
                 url: str) -> dict:
        data = requests.get(url)
        data = data.json()
        return data

    def test_connection(self) -> Dict:
        return self.get_data(FetchRealTimeData.API_URL + FetchRealTimeData.TEST_API)

    def get_price_url(self) -> str:
        return FetchRealTimeData.API_URL + FetchRealTimeData.PRICE_URL.format(symbol=self.symbol)

    def get_order_book_url(self) -> str:
        return FetchRealTimeData.API_URL + FetchRealTimeData.ORDER_BOOK.format(symbol=self.symbol,
                                                                               limit=self.trade_book_limit)

    def generate_data_parallel(self,
                               list_url: List[str],
                               n_jobs: int = -1) -> List[Dict]:
        return Parallel(n_jobs=n_jobs)(delayed(self.get_data)(url) for url in list_url)



if __name__ == "__main__":
    obj = FetchRealTimeData()
    data = obj.generate_data_parallel([obj.get_price_url(), obj.get_order_book_url()])
