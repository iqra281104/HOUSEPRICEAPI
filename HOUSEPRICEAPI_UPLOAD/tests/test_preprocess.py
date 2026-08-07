from src.preprocess import load_data, encode_data, split_data

print("Loading dataset...")
df = load_data()

print(df.head())

print("\nEncoding dataset...")
df = encode_data(df)

print(df.head())

print("\nSplitting dataset...")
X_train, X_test, y_train, y_test = split_data(df)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)

print("\n✅ Preprocessing Successful!")