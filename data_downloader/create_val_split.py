import csv
import os


def main():
    base = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "benchmark_file_lists")
    )
    val_txt = os.path.join(base, "val_scenes.txt")
    input_csv = os.path.join(base, "train_val_set.csv")
    output_csv = os.path.join(base, "val_set.csv")

    # 1. load val IDs
    with open(val_txt, "r") as f:
        val_ids = set(line.strip() for line in f if line.strip())

    # 2. filter train_val_set rows
    with open(input_csv, newline="") as rf, open(output_csv, "w", newline="") as wf:
        reader = csv.reader(rf)
        writer = csv.writer(wf)
        header = next(reader)
        writer.writerow(header)
        for row in reader:
            if row[0] in val_ids:
                writer.writerow(row)


if __name__ == "__main__":
    main()
