import pickle

PICKLE_PATH = "../processed_data.pkl"  # Update path if needed

with open(PICKLE_PATH, "rb") as f:
    data = pickle.load(f)

print(f"✅ Data Type: {type(data)}")
if isinstance(data, tuple):
    print(f"✅ Number of elements: {len(data)}")
    for i, d in enumerate(data):
        print(f"✅ Element {i + 1} Type: {type(d)}")
else:
    print("❌ Unexpected data format!")
