# Quickstart

This example trains `RNRNClassifier` on a tiny synthetic dataset and prints predictions and an explanation for one sample.

```python
import numpy as np
import pandas as pd

# Adjust import to your public API:
from nrn.sklogic.classifiers.RNRNClassifier import RNRNClassifier

# Toy data: 1k samples, 20 features
N, D = 1000, 20
rng = np.random.default_rng(42)
X = rng.normal(size=(N, D)).astype(np.float32)
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(D)])
y_df = pd.DataFrame(y, columns=["target"])

clf = RNRNClassifier(
    layer_sizes=[64, 64],
    epochs=10,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=1e-3,
)

clf.fit(X_df, y_df)

proba = clf.predict_proba(X_df.iloc[:5])
pred  = clf.predict(X_df.iloc[:5])
print("proba:\n", proba.head())
print("pred:\n", pred.head())

text = clf.explain_sample(X_df.iloc[:10], sample_index=0, quantile=1.0)
print("\nEXPLANATION:\n", text)
```

Tips
	•	Prefer passing DataFrames so feature names appear in outputs/explanations.
	•	If training never plateaus, pruning knobs won’t fire (they’re conditional).
	•	For GPU, keep pin_memory=True and consider num_workers>0 on Linux.