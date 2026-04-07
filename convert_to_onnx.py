# -*- coding: utf-8 -*-
"""
Script to convert the Thai W2P numpy checkpoint to ONNX models.

This creates two ONNX models:
  - thai_w2p_encoder.onnx  : encodes a sequence of grapheme indices → last hidden state
  - thai_w2p_decoder.onnx  : single decoder step (token + hidden) → logits + new hidden

Usage:
    python convert_to_onnx.py [--checkpoint logs-run/checkpoint.npy] [--output_dir .]
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal PyTorch modules (inference-only, single-sample)
# ---------------------------------------------------------------------------

class EncoderONNX(nn.Module):
    """Encoder: char indices → last GRU hidden state (batch=1)."""

    def __init__(self, emb_weight, w_ih, w_hh, b_ih, b_hh):
        super().__init__()
        vocab_size, emb_units = emb_weight.shape
        hidden_units = w_hh.shape[1]

        self.emb = nn.Embedding(vocab_size, emb_units)
        self.rnn = nn.GRU(emb_units, hidden_units, batch_first=True)

        with torch.no_grad():
            self.emb.weight.copy_(torch.from_numpy(emb_weight))
            self.rnn.weight_ih_l0.copy_(torch.from_numpy(w_ih))
            self.rnn.weight_hh_l0.copy_(torch.from_numpy(w_hh))
            self.rnn.bias_ih_l0.copy_(torch.from_numpy(b_ih))
            self.rnn.bias_hh_l0.copy_(torch.from_numpy(b_hh))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: int64 tensor of shape (1, seq_len) – grapheme indices.
        Returns:
            last_hidden: float32 tensor of shape (1, hidden_units).
        """
        emb = self.emb(x)                        # (1, seq_len, emb_units)
        _, h_n = self.rnn(emb)                   # h_n: (1, 1, hidden_units)
        return h_n.squeeze(0)                    # (1, hidden_units)


class DecoderStepONNX(nn.Module):
    """One decoder step: (token, hidden) → (logits, new_hidden)."""

    def __init__(self, emb_weight, w_ih, w_hh, b_ih, b_hh, fc_w, fc_b):
        super().__init__()
        vocab_size, emb_units = emb_weight.shape
        hidden_units = w_hh.shape[1]
        output_size = fc_w.shape[0]

        self.emb = nn.Embedding(vocab_size, emb_units)
        self.rnn = nn.GRU(emb_units, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, output_size)

        with torch.no_grad():
            self.emb.weight.copy_(torch.from_numpy(emb_weight))
            self.rnn.weight_ih_l0.copy_(torch.from_numpy(w_ih))
            self.rnn.weight_hh_l0.copy_(torch.from_numpy(w_hh))
            self.rnn.bias_ih_l0.copy_(torch.from_numpy(b_ih))
            self.rnn.bias_hh_l0.copy_(torch.from_numpy(b_hh))
            self.fc.weight.copy_(torch.from_numpy(fc_w))
            self.fc.bias.copy_(torch.from_numpy(fc_b))

    def forward(
        self,
        token: torch.Tensor,
        h: torch.Tensor,
    ):
        """
        Args:
            token: int64 tensor of shape (1, 1) – current decoder token.
            h:     float32 tensor of shape (1, 1, hidden_units) – hidden state.
        Returns:
            logits:  float32 tensor of shape (1, 1, output_size).
            new_h:   float32 tensor of shape (1, 1, hidden_units).
        """
        emb = self.emb(token)                    # (1, 1, emb_units)
        out, new_h = self.rnn(emb, h)            # out: (1,1,hidden), new_h: (1,1,hidden)
        logits = self.fc(out)                    # (1, 1, output_size)
        return logits, new_h


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_encoder(model: EncoderONNX, output_path: str) -> None:
    model.eval()
    # Dummy input: one word of length 3
    dummy_x = torch.zeros(1, 3, dtype=torch.int64)
    torch.onnx.export(
        model,
        (dummy_x,),
        output_path,
        input_names=["x"],
        output_names=["last_hidden"],
        dynamic_axes={
            "x": {1: "seq_len"},
        },
        opset_version=14,
        dynamo=False,
    )
    print(f"Encoder exported → {output_path}")


def export_decoder_step(model: DecoderStepONNX, output_path: str) -> None:
    model.eval()
    hidden_units = model.rnn.hidden_size
    dummy_token = torch.zeros(1, 1, dtype=torch.int64)
    dummy_h = torch.zeros(1, 1, hidden_units, dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy_token, dummy_h),
        output_path,
        input_names=["token", "h"],
        output_names=["logits", "new_h"],
        opset_version=14,
        dynamo=False,
    )
    print(f"Decoder step exported → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert Thai W2P checkpoint to ONNX")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(os.path.dirname(__file__), "logs-run", "checkpoint.npy"),
        help="Path to the numpy checkpoint file (default: logs-run/checkpoint.npy)",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.dirname(__file__),
        help="Directory to save the ONNX models (default: same folder as this script)",
    )
    args = parser.parse_args()

    # Load weights
    variables = np.load(args.checkpoint, allow_pickle=True).item()

    enc_emb   = variables["encoder.emb.weight"]
    enc_w_ih  = variables["encoder.rnn.weight_ih_l0"]
    enc_w_hh  = variables["encoder.rnn.weight_hh_l0"]
    enc_b_ih  = variables["encoder.rnn.bias_ih_l0"]
    enc_b_hh  = variables["encoder.rnn.bias_hh_l0"]

    dec_emb   = variables["decoder.emb.weight"]
    dec_w_ih  = variables["decoder.rnn.weight_ih_l0"]
    dec_w_hh  = variables["decoder.rnn.weight_hh_l0"]
    dec_b_ih  = variables["decoder.rnn.bias_ih_l0"]
    dec_b_hh  = variables["decoder.rnn.bias_hh_l0"]
    fc_w      = variables["decoder.fc.weight"]
    fc_b      = variables["decoder.fc.bias"]

    os.makedirs(args.output_dir, exist_ok=True)

    # Build and export encoder
    encoder = EncoderONNX(enc_emb, enc_w_ih, enc_w_hh, enc_b_ih, enc_b_hh)
    export_encoder(encoder, os.path.join(args.output_dir, "thai_w2p_encoder.onnx"))

    # Build and export decoder step
    decoder = DecoderStepONNX(dec_emb, dec_w_ih, dec_w_hh, dec_b_ih, dec_b_hh, fc_w, fc_b)
    export_decoder_step(decoder, os.path.join(args.output_dir, "thai_w2p_decoder.onnx"))

    print("Done.")


if __name__ == "__main__":
    main()
