from argparse import ArgumentParser
import torch

from inference import get_default_model
from train import get_tokens, load_eval_data, eval
from tokenizer import BPETokenizer, END_OF_TEXT


def eval_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_path", required=True, help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--val_data", required=True, help="Path to the validation data file"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_default_model(device=torch.device(device))

    checkpoint_path = args.checkpoint_path
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = BPETokenizer(vocab={}, merges=[], special_tokens=[END_OF_TEXT])
    val_data = get_tokens(args.val_data, tokenizer)
    val_input_ids, val_target_ids = load_eval_data(
        x=val_data,
        eval_batch_size=128,
        context_length=256,
        device=device,
    )
    val_loss = eval(model, val_input_ids, val_target_ids)
    print(f"Validation Loss: {val_loss}")


if __name__ == "__main__":
    eval_main()
