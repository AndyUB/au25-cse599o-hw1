from argparse import ArgumentParser
import torch

from tokenizer import END_OF_TEXT, BPETokenizer
from transformer import Transformer, softmax


def decode(
    model: Transformer,
    tokenizer: BPETokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: torch.device | None = None,
) -> str:
    END_OF_TEXT_ID = tokenizer.encode(END_OF_TEXT)[0]
    tokens = (
        torch.tensor(tokenizer.encode(prompt), device=device).long().unsqueeze(0)
    )  # (1, seq_len)
    for _ in range(max_tokens):
        logits: torch.Tensor = model(tokens)  # (1, seq_len, vocab_size)
        logits = torch.squeeze(logits, 0)  # (seq_len, vocab_size)
        logits = logits[-1, :]  # (vocab_size,)

        logits = logits / temperature
        probs = softmax(logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_prob = 0
        cutoff_index = 0
        for i in range(sorted_probs.shape[0]):
            cumulative_prob += sorted_probs[i].item()
            cutoff_index += 1
            if cumulative_prob >= top_p:
                break
        if cutoff_index < sorted_probs.shape[0]:
            sorted_probs[cutoff_index:] = 0.0
        sorted_probs = sorted_probs / torch.sum(sorted_probs)

        next_token = torch.multinomial(sorted_probs, 1)
        next_token = sorted_indices[next_token]

        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == END_OF_TEXT_ID:
            break

    return tokenizer.decode(tokens.squeeze(0).tolist())


def get_default_model(device: torch.device) -> Transformer:
    model = Transformer(
        vocab_size=50257,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        context_length=256,
        theta=10000,
        device=torch.device(device),
    )
    return model


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Prompt text for generation"
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling threshold",
    )

    args = parser.parse_args()
    print(f"Arguments: {args}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_default_model(device=torch.device(device))

    checkpoint = torch.load(args.model_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = BPETokenizer(vocab={}, merges=[], special_tokens=[END_OF_TEXT])
    generated_text = decode(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=torch.device(device),
    )
    print(generated_text)


if __name__ == "__main__":
    main()
