import pickle
import gzip
import os

def create_full_test_labels():
    # Load the full dataset
    data_path = 'vn_sentence_data/sentence_vn.pkl'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return

    print(f"Loading {data_path}...")
    with open(data_path, 'rb') as f:
        full_data = pickle.load(f)
    
    print(f"Total samples: {len(full_data)}")
    
    # Get all keys
    keys = list(full_data.keys())
    
    # Take the last 63 samples for testing
    test_keys = keys[-63:]
    test_data = {k: full_data[k] for k in test_keys}
    
    print(f"Created test set with {len(test_data)} samples")
    print(f"First 5 test keys: {test_keys[:5]}")
    
    # Save to data/VSL/labels.test
    output_path = 'data/VSL/labels.test'
    print(f"Saving to {output_path}...")
    
    with gzip.open(output_path, 'wb') as f:
        pickle.dump(test_data, f)
        
    print("Done!")

if __name__ == "__main__":
    create_full_test_labels()