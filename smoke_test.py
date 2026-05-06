"""
Smoke test: kiểm tra model forward/backward không bị lỗi.
Chạy: python smoke_test.py
"""
import argparse
import torch
from model import DecoderLM


def make_config(**overrides):
    defaults = dict(
        vocab_size=50000, pad_token_id=1, bos_token_id=0, eos_token_id=2,
        max_seq_len=64, d_model=128, n_heads=4, n_layers=2, d_ff=256,
        dropout=0.0, squeeze_ratio=4,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_batch(B, T, vocab_size, device):
    input_ids = torch.randint(3, vocab_size, (B, T), device=device)
    labels    = torch.randint(3, vocab_size, (B, T), device=device)
    mask      = torch.ones(B, T, dtype=torch.bool, device=device)
    return input_ids, labels, mask


def test_lm_forward(device):
    cfg   = make_config()
    model = DecoderLM(cfg).to(device).eval()
    ids, labels, mask = _make_batch(2, 32, cfg.vocab_size, device)

    with torch.no_grad():
        logits = model(ids, attention_mask=mask)
    assert logits.shape == (2, 32, cfg.vocab_size), f"bad logits shape: {logits.shape}"
    print("  [PASS] lm_forward: logits shape OK")


def test_lm_loss(device):
    cfg   = make_config()
    model = DecoderLM(cfg).to(device)
    ids, labels, mask = _make_batch(2, 32, cfg.vocab_size, device)

    loss = model(ids, attention_mask=mask, labels=labels)
    assert loss.ndim == 0, "loss should be scalar"
    assert not torch.isnan(loss), "loss is NaN"
    loss.backward()
    print(f"  [PASS] lm_loss: loss={loss.item():.4f}, backward OK")


def test_cross_attention(device):
    cfg   = make_config()
    model = DecoderLM(cfg).to(device).eval()
    ids, labels, mask = _make_batch(2, 32, cfg.vocab_size, device)
    encoder_out = torch.randn(2, 16, cfg.d_model, device=device)

    with torch.no_grad():
        logits = model(ids, attention_mask=mask, encoder_output=encoder_out)
    assert logits.shape == (2, 32, cfg.vocab_size)
    print("  [PASS] cross_attention: logits shape OK")


def test_variable_seqlen(device):
    cfg   = make_config()
    model = DecoderLM(cfg).to(device).eval()

    for T in [1, 7, 33, 63]:
        ids, _, mask = _make_batch(1, T, cfg.vocab_size, device)
        with torch.no_grad():
            logits = model(ids, attention_mask=mask)
        assert logits.shape == (1, T, cfg.vocab_size)
    print("  [PASS] variable_seqlen: T=1,7,33,63 OK")


def test_padded_batch(device):
    cfg   = make_config()
    model = DecoderLM(cfg).to(device).eval()
    B, T  = 3, 32
    ids   = torch.randint(3, cfg.vocab_size, (B, T), device=device)
    # simulate padding: last 8 tokens of row 0 are pad
    mask  = torch.ones(B, T, dtype=torch.bool, device=device)
    mask[0, 24:] = False

    with torch.no_grad():
        logits = model(ids, attention_mask=mask)
    assert logits.shape == (B, T, cfg.vocab_size)
    print("  [PASS] padded_batch OK")


def test_squeeze_ratio(device):
    for r in [1, 2, 4, 8]:
        cfg   = make_config(squeeze_ratio=r)
        model = DecoderLM(cfg).to(device).eval()
        ids, _, mask = _make_batch(2, 32, cfg.vocab_size, device)
        with torch.no_grad():
            logits = model(ids, attention_mask=mask)
        assert logits.shape == (2, 32, cfg.vocab_size)
    print("  [PASS] squeeze_ratio=1,2,4,8 OK")


def run_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    tests = [
        test_lm_forward,
        test_lm_loss,
        test_cross_attention,
        test_variable_seqlen,
        test_padded_batch,
        test_squeeze_ratio,
    ]

    passed, failed = 0, 0
    for fn in tests:
        print(f"[{fn.__name__}]")
        try:
            fn(device)
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
