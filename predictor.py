import numpy as np

# Must be the exact column order used during training (after get_dummies)
FEATURES = [
    "age", "height", "weight", "waistline",
    "sight_left", "sight_right", "hear_left", "hear_right",
    "SBP", "DBP", "BLDS",
    "tot_chole", "HDL_chole", "LDL_chole", "triglyceride",
    "hemoglobin", "urine_protein", "serum_creatinine",
    "SGOT_AST", "SGOT_ALT", "gamma_GTP",
    "SMK_stat_type_cd", "sex_Female", "sex_Male",
]


class Predictor:
    """Pure-numpy inference for the trained ANN (Dropout is inactive at inference)."""

    def __init__(self, path: str):
        data = np.load(path)
        self.mean = data["mean"].astype(np.float32)
        self.scale = data["scale"].astype(np.float32)
        n_layers = len([k for k in data.files if k.startswith("W")])
        self.layers = [
            (data[f"W{i}"].astype(np.float32), data[f"b{i}"].astype(np.float32))
            for i in range(n_layers)
        ]
        if len(self.mean) != len(FEATURES):
            raise ValueError(
                f"Model expects {len(self.mean)} features but FEATURES has {len(FEATURES)}"
            )

    def predict_proba(self, x) -> float:
        """x: array of shape (1, n_features) in FEATURES order. Returns P(drinker)."""
        x = (np.asarray(x, dtype=np.float32) - self.mean) / self.scale
        last = len(self.layers) - 1
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < last:
                x = np.maximum(x, 0.0)  # ReLU
        logit = float(x.ravel()[0])
        return float(1.0 / (1.0 + np.exp(-logit)))  # sigmoid
