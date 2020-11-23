from utils import MLMDataset

dataset = MLMDataset('glue_data', 'cola', 120, 'bert-base-uncased')
print()