import pickle
import gzip
import os

def inspect_gzip_pickle(path):
    print(f"Inspecting {path}...")
    try:
        with gzip.open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"  Type: {type(data)}")
        if isinstance(data, dict):
            print(f"  Length: {len(data)}")
            print(f"  Keys (first 5): {list(data.keys())[:5]}")
        elif isinstance(data, list):
            print(f"  Length: {len(data)}")
    except Exception as e:
        print(f"  Error reading as gzip: {e}")
        # Try reading as normal pickle
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            print(f"  Read as normal pickle.")
            print(f"  Type: {type(data)}")
            if isinstance(data, dict):
                print(f"  Length: {len(data)}")
                print(f"  Keys (first 5): {list(data.keys())[:5]}")
            elif isinstance(data, list):
                print(f"  Length: {len(data)}")
        except Exception as e2:
            print(f"  Error reading as normal pickle: {e2}")

def inspect_pickle(path):
    print(f"Inspecting {path}...")
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"  Type: {type(data)}")
        if isinstance(data, dict):
            print(f"  Length: {len(data)}")
            print(f"  Keys (first 5): {list(data.keys())[:5]}")
        elif isinstance(data, list):
            print(f"  Length: {len(data)}")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    if os.path.exists('data/VSL/labels.test'):
        inspect_gzip_pickle('data/VSL/labels.test')
    else:
        print("data/VSL/labels.test not found")

    if os.path.exists('vn_sentence_data/vn_sentence_data.file'):
        print(f"Inspecting vn_sentence_data/vn_sentence_data.file...")
        try:
            with gzip.open('vn_sentence_data/vn_sentence_data.file', 'rb') as f:
                data = pickle.load(f)
            print(f"  Type: {type(data)}")
            if isinstance(data, list):
                print(f"  Length: {len(data)}")
                print(f"  First 5 items: {data[:5]}")
            elif isinstance(data, dict):
                print(f"  Length: {len(data)}")
                print(f"  Keys (first 5): {list(data.keys())[:5]}")
        except Exception as e:
            print(f"  Error reading as gzip pickle: {e}")
    else:
        print("vn_sentence_data/vn_sentence_data.file not found")