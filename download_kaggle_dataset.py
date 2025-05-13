import kagglehub
import os

# Download the dataset
path = kagglehub.dataset_download("nelgiriyewithana/world-stock-prices-daily-updating")
print("Path to dataset files:", path)

# List the contents of the downloaded dataset directory
print("\nContents of the dataset directory:")
for root, dirs, files in os.walk(path):
    for file in files:
        print(os.path.join(root, file))