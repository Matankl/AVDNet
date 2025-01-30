import os
import torch
from optimization import evaluate_on_test

if __name__ == '__main__':
    test_folder = r"D:\Database\Audio\DeepFakeProject\Fake\Raw generated Data\fake database\generated_audio\4 sec Processed"
    csv_files = [os.path.join(test_folder, csv) for csv in os.listdir(test_folder) if csv.endswith(".csv")]
    # csv_files = [r"C:\Users\parde\PycharmProjects\The-model\data\Inputs\test_30h.csv"]

    model = torch.load(
        r"Model Demo/DeepFakeModel_lr=1.0995252939300183e-05_bs=16_drop=0.65_layers=5_valloss=0.1127.pth")

    results = []
    for csv_file in csv_files:
        print(f"currently processing: {csv_file}")
        result = evaluate_on_test(model, csv_file, 16)
        results.append((result, csv_file))

    print(*results, sep = "\n")