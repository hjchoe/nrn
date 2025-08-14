# Installation

## Prerequisites
- Python 3.10+ (recommended)
- pip ≥ 22
- (Optional) CUDA toolkit if using GPU

## From PyPI
```bash
pip install centaur-nrn
```

## From source (development)
```bash
git clone https://github.com/centaurinstitute/nrn.git
cd nrn
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e .[dev]
```

## Verify install
```bash
python -c "import nrn, sys; print('nrn version OK'); sys.exit(0)"
```

If you see import errors, ensure you ran pip install -e . from the repo root, not a subfolder.
