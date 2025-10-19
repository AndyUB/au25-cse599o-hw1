from argparse import ArgumentParser
import os
import torch
import numpy as np

from tokenizer import BPETokenizer
from transformer import Transformer
from training import AdamW, load_data, lr_cosine_schedule, load_checkpoint, save_checkpoint

END_OF_TEXT = "<|endoftext|>"


def get_tokens(data_path: str, tokenizer: BPETokenizer) -> np.ndarray:
    """
    Load data from a text file and convert it to a numpy array of token IDs.

    Args:
        data_path (str): Path to the text file containing the data.
        tokenizer (BPETokenizer): Tokenizer to encode the text.

    Returns:
        np.ndarray: Numpy array of token IDs.
    """
    bin_path = data_path + ".bin"
    if not os.path.exists(bin_path):
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = tokenizer.encode(text)
        np.array(tokens, dtype=np.int32).tofile(bin_path)
    return np.memmap(bin_path, dtype=np.int32, mode="r")


def main():
    parser = ArgumentParser(description="Train a transformer model")
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to training data"
    )
    parser.add_argument(
        "--val_data", type=str, required=True, help="Path to validation data"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1000,
        help="Number of iterations between saving checkpoints",
    )
    parser.add_argument(
        "--checkpoint_resume_path",
        type=str,
        default="",
        help="Path to a checkpoint to resume training from",
    )

    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of transformer blocks"
    )
    parser.add_argument(
        "--d_model", type=int, default=512, help="Model's hidden dimension"
    )
    parser.add_argument(
        "--num_heads", type=int, default=16, help="Number of attention heads"
    )
    parser.add_argument(
        "--d_ff", type=int, default=1344, help="Feedforward network dimension"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=256, help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5000, help="Number of training epochs"
    )

    parser.add_argument(
        "--rope_theta", type=float, default=10000, help="RoPE theta parameter"
    )

    parser.add_argument(
        "--adamw_beta1", type=float, default=0.9, help="AdamW beta1 parameter"
    )
    parser.add_argument(
        "--adamw_beta2", type=float, default=0.999, help="AdamW beta2 parameter"
    )
    parser.add_argument(
        "--adamw_eps", type=float, default=1e-8, help="AdamW epsilon parameter"
    )
    parser.add_argument(
        "--adamw_weight_decay", type=float, default=0.01, help="Weight decay for AdamW"
    )

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--lr_max", type=float, default=1e-3, help="Maximum learning rate"
    )
    parser.add_argument(
        "--lr_min", type=float, default=1e-4, help="Minimum learning rate"
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=500, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--cosine_iters",
        type=int,
        default=4000,
        help="Number of cosine annealing iterations",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )

    parser.add_argument(
        "--seed", type=int, default=599, help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("loading data...")
    tokenizer = BPETokenizer(vocab={}, merges=[], special_tokens=[END_OF_TEXT])
    train_data = get_tokens(args.train_data, tokenizer)
    val_data = get_tokens(args.val_data, tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")
    model = Transformer(
        vocab_size=50257,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        context_length=args.max_seq_length,
        theta=args.rope_theta,
        device=torch.device(device),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adamw_beta1, args.adamw_beta2),
        eps=args.adamw_eps,
        weight_decay=args.adamw_weight_decay,
    )
    if args.checkpoint_resume_path:
        start_epoch = load_checkpoint(
            args.checkpoint_resume_path,
            model,
            optimizer,
        )
        print(f"Resuming training from checkpoint {args.checkpoint_resume_path} at epoch {start_epoch}")
    else:
        start_epoch = 0
        print("Starting training from epoch 0")

    for epoch in range(start_epoch, args.num_epochs):
        input_ids, target_ids = load_data(
            x=train_data,
            batch_size=args.batch_size,
            context_length=args.max_seq_length,
            device=device,
        )

        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"
            )
            save_checkpoint(checkpoint_path, model, optimizer, epoch + 1)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()
