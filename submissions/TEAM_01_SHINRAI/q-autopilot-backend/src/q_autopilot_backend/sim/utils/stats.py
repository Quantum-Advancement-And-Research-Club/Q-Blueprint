import numpy as np


def sd(data: np.ndarray) -> float:
    """Standard deviation."""
    return float(np.std(data, ddof=1))


def se(data: np.ndarray) -> float:
    """Standard error of the mean."""
    return float(np.std(data, ddof=1) / np.sqrt(len(data)))


def ci95(data: np.ndarray) -> float:
    """95% confidence interval half-width."""
    return float(1.96 * np.std(data, ddof=1) / np.sqrt(len(data)))


def uncertainty(data: np.ndarray, utype: str = "SD") -> float:
    """Compute uncertainty based on type."""
    if utype == "SD":
        return sd(data)
    elif utype == "SE":
        return se(data)
    elif utype == "CI95":
        return ci95(data)
    else:
        raise ValueError(f"Unknown uncertainty type: {utype}")