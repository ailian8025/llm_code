from ds.util import load_json_data, save_json_data

if __name__ == '__main__':
    path = "D:\datasets\Alpaca-CoT\Vicuna.json"
    data = load_json_data(path)
    # save_json_data(data[:5000], "D:\datasets\Alpaca-CoT\Vicuna_5000.json")
    save_json_data(data[:500], "D:\datasets\Alpaca-CoT\Vicuna_500.json")