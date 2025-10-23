from argparse import ArgumentParser
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tokenizer import BPETokenizer, END_OF_TEXT
from transformer import Transformer
from util import (
    AdamW,
    load_data,
    lr_cosine_schedule,
    load_checkpoint,
    save_checkpoint,
    cross_entropy_loss,
    gradient_clipping,
)


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


def load_eval_data(
    x: np.ndarray,
    eval_batch_size: int,
    context_length: int,
    device: str,
    num_batches: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Load a batch of data for evaluation.

    Args:
        x: Numpy array of token IDs.
        eval_batch_size (int): Number of sequences in an evaluation batch.
        context_length (int): Length of each sequence.
        device (str): Device to load the data onto.
        num_batches (int | None): Number of batches to load. If None,
            defaults to x.shape[0] // (eval_batch_size * context_length).

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor]]: A tuple containing:
            - input_ids: List of num_batches tensors of shape
                (eval_batch_size, context_length).
            - target_ids: List of num_batches tensors of shape
                (eval_batch_size, context_length).
    """
    num_tokens = x.shape[0]
    device = torch.device(device)
    if num_batches is None:
        num_batches = num_tokens // (eval_batch_size * context_length)
    input_ids_list = []
    target_ids_list = []
    for i in range(num_batches):
        start = i * eval_batch_size * context_length
        input_ids = (
            torch.stack(
                [
                    torch.from_numpy(
                        x[
                            start
                            + j * context_length : start
                            + (j + 1) * context_length
                        ].copy()
                    )
                    for j in range(eval_batch_size)
                ]
            )
            .long()
            .to(device)
        )
        target_ids = (
            torch.stack(
                [
                    torch.from_numpy(
                        x[
                            start
                            + j * context_length
                            + 1 : start
                            + (j + 1) * context_length
                            + 1
                        ].copy()
                    )
                    for j in range(eval_batch_size)
                ]
            )
            .long()
            .to(device)
        )
        input_ids_list.append(input_ids)
        target_ids_list.append(target_ids)
    return input_ids_list, target_ids_list


def eval(
    model: Transformer,
    val_input_ids: list[torch.Tensor],
    val_target_ids: list[torch.Tensor],
) -> float:
    """
    Evaluate the model on the validation dataset.

    Args:
        model (Transformer): The transformer model to evaluate.
        val_input_ids (list[torch.Tensor]): List of input ID tensors.
        val_target_ids (list[torch.Tensor]): List of target ID tensors.

    Returns:
        float: The average validation loss.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for input_ids, target_ids in zip(val_input_ids, val_target_ids):
            batch_logits = model(input_ids)
            batch_loss = cross_entropy_loss(batch_logits, target_ids)
            total_loss += batch_loss.item()
    val_loss = total_loss / len(val_input_ids)
    return val_loss


def save_loss_map(
    loss_map: dict[int, tuple[float, float]],
    filepath: str,
) -> None:
    """
    Save the training and validation loss map to a file.

    Args:
        loss_map (dict[int, tuple[float, float]]): A dictionary mapping
            epoch numbers to (training loss, validation loss) tuples.
        filepath (str): The path to save the loss map file.
    """
    with open(filepath, "w") as f:
        f.write("Epoch,Training Loss,Validation Loss\n")
        for epoch, (train_loss, val_loss) in loss_map.items():
            f.write(f"{epoch},{train_loss},{val_loss}\n")


def plot_loss(loss_map: dict[int, tuple[float, float]], filepath: str) -> None:
    """
    Plot the training and validation loss curves and save to a file.

    Args:
        loss_map (dict[int, tuple[float, float]]): A dictionary mapping
            epoch numbers to (training loss, validation loss) tuples.
        filepath (str): The path to save the plot file.
    """
    epochs = sorted(list(loss_map.keys()))
    train_losses = [loss[0] for loss in loss_map.values()]
    val_losses = [loss[1] for loss in loss_map.values()]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(filepath)
    plt.close()


def main():
    parser = ArgumentParser(description="Train a transformer model")
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to training data"
    )
    parser.add_argument(
        "--val_data", type=str, required=True, help="Path to validation data"
    )
    parser.add_argument("--log_dir", type=str, required=True, help="Directory for logs")
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
        "--eval_interval", type=int, default=100, help="Evaluation interval"
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
        "--enable_lr_schedule", action="store_true", help="Enable LR scheduling"
    )
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
    print(f"Arguments: {args}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    print("loading data...")
    tokenizer = BPETokenizer(vocab={}, merges=[], special_tokens=[END_OF_TEXT])
    train_data = get_tokens(args.train_data, tokenizer)
    val_data = get_tokens(args.val_data, tokenizer)
    val_input_ids, val_target_ids = load_eval_data(
        x=val_data,
        eval_batch_size=args.batch_size,
        context_length=args.max_seq_length,
        device=device,
        num_batches=10,
    )

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
        print(
            f"Resuming training from checkpoint {args.checkpoint_resume_path} at epoch {start_epoch}"
        )
    else:
        start_epoch = 0
        print("Starting training from epoch 0")

    loss_map = {}

    for epoch in range(start_epoch, args.num_epochs):
        input_ids, target_ids = load_data(
            x=train_data,
            batch_size=args.batch_size,
            context_length=args.max_seq_length,
            device=device,
        )
        logits = model(input_ids)
        loss = cross_entropy_loss(logits, target_ids)
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(
            parameters=list(model.parameters()), max_norm=args.max_grad_norm
        )
        if args.enable_lr_schedule:
            lr = lr_cosine_schedule(
                epoch,
                lr_max=args.lr_max,
                lr_min=args.lr_min,
                warmup_iters=args.warmup_iters,
                cosine_iters=args.cosine_iters,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        optimizer.step()

        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"ckpt{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        if (epoch + 1) % args.eval_interval == 0:
            val_loss = eval(model, val_input_ids, val_target_ids)
            print(
                f"Epoch {epoch+1}, "
                f"Training Loss: {loss.item():.4f}, "
                f"Validation Loss: {val_loss:.4f}"
            )
            model.train()
            loss_map[epoch + 1] = (loss.item(), val_loss)

    loss_map_path = os.path.join(args.log_dir, "loss_map.csv")
    save_loss_map(loss_map, loss_map_path)
    loss_plot_path = os.path.join(args.log_dir, "loss_plot.png")
    plot_loss(loss_map, loss_plot_path)


if __name__ == "__main__":
    main()
