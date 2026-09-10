from __future__ import annotations

from pathlib import Path
import pickle
import joblib


def save_model_artifacts(model, output_dir: str = ".", base_name: str = "ddos_model") -> dict[str, str]:
    """
    Save an already-trained model in both Joblib and Pickle formats.

    Usage from notebook after training:
        from model_export import save_model_artifacts
        save_model_artifacts(rf_tuned)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib_path = out_dir / f"{base_name}.joblib"
    pkl_path = out_dir / f"{base_name}.pkl"

    joblib.dump(model, joblib_path)
    with pkl_path.open("wb") as f:
        pickle.dump(model, f)

    return {"joblib": str(joblib_path), "pkl": str(pkl_path)}
