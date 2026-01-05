import argparse
import glob
import os
import shutil
import cocopp


def merge_and_process_batches(pattern: str, output_folder: str) -> None:
    batch_folders = glob.glob(pattern)

    if not batch_folders:
        print(f"Didn't find files matchin pattern: {pattern}")
        return

    target_dir = os.path.join("exdata", output_folder)

    if os.path.exists(target_dir):
        print(f"Output folder {target_dir} already exists")
        return

    os.makedirs(target_dir)

    print(f"Merging batches {len(batch_folders)} to {target_dir}...")
    for folder in batch_folders:
        print(f"Copying results from {folder}")
        for item in os.listdir(folder):
            source_path = os.path.join(folder, item)
            target_path = os.path.join(target_dir, item)

            if os.path.isdir(source_path):
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
                for subitem in os.listdir(source_path):
                    shutil.copy(os.path.join(source_path, subitem), os.path.join(target_path, subitem))
            else:
                shutil.copy(source_path, target_path)

    print("Merging done. Running cocopp...")

    cocopp.main(target_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge batches folders and run cocopp"
    )

    parser.add_argument(
        "pattern", type=str, help="Pattern of input files (eg. 'exdata/alg_*')"
    )
    parser.add_argument("output", type=str, help="Output folder name")

    args = parser.parse_args()

    merge_and_process_batches(args.pattern, args.output)
