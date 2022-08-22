from DataFetch.FetchRealTimeData import FetchRealTimeData
import json
import time
import tqdm
from datetime import datetime

def generate_data(symbol: str,
                  trade_book_limit: int = 2,
                  quantity_of_data_to_fetch: int = 100,
                  path_to_save_data: str = '',
                  save_file: bool = True) -> None:
    fech_real_time_data = FetchRealTimeData(symbol=symbol,
                                            trade_book_limit=trade_book_limit)
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    urls = [fech_real_time_data.get_price_url(), fech_real_time_data.get_order_book_url()]
    price_execution_data_list = []
    trade_book_data_list = []
    for _ in tqdm.tqdm(range(quantity_of_data_to_fetch)):
        start_time = time.time()
        # fetch data
        price_execution_data = fech_real_time_data.get_data(urls[0])
        trade_book_data = fech_real_time_data.get_data(urls[1])
        price_execution_data["time"] = start_time
        price_execution_data["delta_time"] = time.time()-start_time
        trade_book_data["time"] = start_time
        trade_book_data["delta_time"] = time.time() - start_time
        price_execution_data_list.append(price_execution_data)
        trade_book_data_list.append(trade_book_data)

    price_execution_data_list = json.dumps(price_execution_data_list, indent=4)
    json_trade_book_data_list = json.dumps(trade_book_data_list, indent=4)
    if save_file:
        # Writing to sample.json
        with open(f"{path_to_save_data}price_execution_data_list_{dt_string}.json", "w") as outfile:
            outfile.write(price_execution_data_list)
        with open(f"{path_to_save_data}trade_book_data_{dt_string}.json", "w") as outfile:
            outfile.write(json_trade_book_data_list)


if __name__ == "__main__":
    generate_data("BTCUSDT", quantity_of_data_to_fetch=1, path_to_save_data = "/Users/efim/PycharmProjects/SimpleAlgoTrade/DataBase/files/")
