import os
import torch
from argparse import ArgumentParser

from inference import get_default_model
from util import AdamW, load_checkpoint


def split_ckpt(ckpt_path: str, output_dir: str) -> None:
    """
    Split a checkpoint file into smaller files.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        output_dir (str): Directory to save the split files.
    """

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(ckpt_path, "rb") as f:
        data = f.read()

    # Split the data into smaller chunks and save them
    chunk_size = 1024 * 1024 * 50  # 50MB
    num_chunks = (len(data) + chunk_size - 1) // chunk_size
    print(f"Splitting {ckpt_path} into {num_chunks} chunks...")

    for i in range(0, len(data), chunk_size):
        lo = i
        hi = min(i + chunk_size, len(data))
        chunk = data[lo:hi]
        chunk_path = os.path.join(output_dir, f"part{i // chunk_size}.bin")
        with open(chunk_path, "wb") as chunk_file:
            chunk_file.write(chunk)
        print(f"Saved chunk {chunk_path}")


def merge_ckpt(input_dir: str, output_path: str) -> None:
    """
    Merge smaller checkpoint files into a single file.

    Args:
        input_dir (str): Directory containing the split files.
        output_path (str): Path to save the merged checkpoint file.
    """

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

    chunk_files = sorted(
        [
            f
            for f in os.listdir(input_dir)
            if f.startswith("part") and f.endswith(".bin")
        ],
        key=lambda x: int(x[4:-4]),
    )
    print(f"Merging {len(chunk_files)} chunks into {output_path}...")

    with open(output_path, "wb") as out_file:
        for chunk_file in chunk_files:
            chunk_path = os.path.join(input_dir, chunk_file)
            with open(chunk_path, "rb") as cf:
                data = cf.read()
                out_file.write(data)
            print(f"Merged chunk {chunk_path}")


def test_split_merge(ckpt_path: str, split_dir: str) -> None:
    """
    Test the split and merge functions.

    Args:
        ckpt_path (str): Path to the original checkpoint file.
        split_dir (str): Directory to save the split files.
    """

    merge_path = ckpt_path + ".merged"

    split_ckpt(ckpt_path, split_dir)
    merge_ckpt(split_dir, merge_path)

    # Verify that the merged file is identical to the original
    with open(ckpt_path, "rb") as f1, open(merge_path, "rb") as f2:
        original_data = f1.read()
        merged_data = f2.read()
        assert original_data == merged_data, "Merged file does not match original!"

    print("Test passed: Merged file matches the original")


def extract_model_from_checkpoint(ckpt_path: str, model_path: str) -> None:
    """
    Extract the model state dictionary from a checkpoint file,
    and save it to the specified model path.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        model_path (str): Path to save the model state dictionary.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist.")

    checkpoint = torch.load(ckpt_path)
    model_state_dict = checkpoint["model_state_dict"]
    torch.save(model_state_dict, model_path)
    print(f"Extracted model state dict saved to {model_path}")

    model_ckpt = get_default_model(device=torch.device("cpu"))
    optimizer = AdamW(
        model_ckpt.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    load_checkpoint(ckpt_path, model_ckpt, optimizer)

    model = get_default_model(device=torch.device("cpu"))
    model.load_state_dict(torch.load(model_path))

    mismatched_params = []
    for name, param in model.named_parameters():
        ckpt_param = model_ckpt.state_dict()[name]
        if not torch.equal(param, ckpt_param):
            mismatched_params.append(name)
    if mismatched_params:
        print(f"Test failed! Mismatched parameters: {mismatched_params}.")
    else:
        print("Test passed! Extracted model matches the checkpoint model.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["split", "merge", "test", "extract_model"],
        required=True,
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to the checkpoint file."
    )
    parser.add_argument(
        "--chunk_dir", type=str, required=False, help="Directory of chunk files"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="Path to save the extracted model state dictionary",
    )
    args = parser.parse_args()

    if args.mode == "extract_model":
        if not args.model_path:
            raise ValueError("--model_path is required for extract_model mode")
    else:
        if not args.chunk_dir:
            raise ValueError(f"--chunk_dir is required for {args.mode} mode")

    if args.mode == "extract_model":
        extract_model_from_checkpoint(args.ckpt_path, args.model_path)
    elif args.mode == "split":
        split_ckpt(args.ckpt_path, args.chunk_dir)
    elif args.mode == "merge":
        merge_ckpt(args.chunk_dir, args.ckpt_path)
    elif args.mode == "test":
        test_split_merge(args.ckpt_path, args.chunk_dir)
