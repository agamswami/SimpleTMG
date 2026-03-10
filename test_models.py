"""Quick verification script: import and forward-pass test for all 3 models."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import argparse

def make_configs():
    args = argparse.Namespace(
        seq_len=96, pred_len=96, output_attention=False,
        use_norm=True, geomattn_dropout=0.5, alpha=1.0,
        kernel_size=None, d_model=32, d_ff=32, e_layers=1,
        enc_in=7, dec_in=7, c_out=7, embed='timeF', freq='h',
        dropout=0.1, requires_grad=True, wv='db1', m=3,
        factor=1, activation='gelu', compile=False
    )
    return args

def test_model(model_name, ModelClass):
    configs = make_configs()
    model = ModelClass(configs).float()
    B, L, N = 2, 96, 7
    x = torch.randn(B, L, N)
    out, attns = model(x, None, None, None)
    assert out.shape == (B, 96, N), f"Expected (2, 96, 7), got {out.shape}"
    print(f"  ✓ {model_name}: output shape = {out.shape}")

if __name__ == '__main__':
    print("Testing model imports and forward passes...\n")
    
    from model.SimpleTM import Model as OrigModel
    test_model("SimpleTM (Original)", OrigModel)
    
    from model.SimpleTM_SWT import Model as SWTModel
    test_model("SimpleTM_SWT", SWTModel)
    
    from model.SimpleTM_FFT import Model as FFTModel
    test_model("SimpleTM_FFT", FFTModel)
    
    print("\n✓ All 3 models passed forward-pass verification!")
