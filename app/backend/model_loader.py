"""
Load the frozen INCLUDE262 V2 model.

The architecture is NOT re-declared here. It is imported verbatim from
scripts/82_train_include262_partaware_htma_v2.py via importlib, exactly the way
scripts 83 and 84 load it. Only `ema_state` is loaded from the checkpoint.
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import config  # noqa: E402


def load_script_module(path, name):
    """Import an existing project script as a module without running its main()."""
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # module-level only: imports + class/const defs
    return module


def load_model(device=None, variant="original"):
    """
    Returns (model, checkpoint_dict).

    model  : scripts/82 -> IncludePartAwareConvHTMAV2, default hyperparameters
             (num_classes=262, d_model=320, num_heads=8, num_layers=4, dropout=0.12),
             weights = checkpoint["ema_state"] (falls back to "model_state"), model.eval().

    variant: "original" (default, the frozen 98.04%-test checkpoint) or
             "webcam_adapted" (webcam/adapt_train.py output — falls back to original
             if that file does not exist yet).
    """
    config.check_paths()
    device = torch.device(device or config.DEVICE)

    ckpt_path = config.checkpoint_for(variant)
    s82 = load_script_module(config.SCRIPT_82, "include262_train_v2")
    model = s82.IncludePartAwareConvHTMAV2()  # exact defaults, matches scripts 83/84

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = checkpoint.get("ema_state") or checkpoint.get("model_state")
    if state is None:
        raise RuntimeError(f"Checkpoint {ckpt_path} has neither 'ema_state' nor 'model_state'.")

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    checkpoint["_path"] = str(ckpt_path)
    checkpoint["_variant"] = variant
    return model, checkpoint


if __name__ == "__main__":
    m, ck = load_model()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Loaded IncludePartAwareConvHTMAV2  |  {n_params:,} params ({n_params / 1e6:.2f} M)")
    print(f"Checkpoint epoch      : {ck.get('epoch')}")
    print(f"Stored best val top-1 : {ck.get('best_top1')}")
    print(f"Weights loaded from   : ema_state")
