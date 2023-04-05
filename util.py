import json


def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    return data


def save_json_data(data, path):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(data, f)


if __name__ == '__main__':
    path = "D:\datasets\Alpaca-CoT\Vicuna.json"
    data = load_json_data(path)
