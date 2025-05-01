# SmartSec Auditor

SmartSec Auditor is an automated vulnerability detection system for Ethereum smart contracts, leveraging advanced Graph Neural Networks (GNNs) to accurately detect potential vulnerabilities. The system integrates blockchain logging for transparent, immutable auditing of detection results.

---

## Overview

SmartSec Auditor uses specialized Graph Neural Network architectures to analyze smart contracts represented as graph structures derived from their Abstract Syntax Trees (ASTs). It detects specific vulnerabilities such as Reentrancy, Timestamp Dependency, and Infinite Loops, logging results to a blockchain for transparency.

## Project Structure

```
smartsec_auditor/
├── blockchain/
│   ├── deploy_logger.py
│   └── log_prediction_to_chain.py
├── data/
│   ├── raw/
│   │   ├── secure/
│   │   └── vulnerable/
│   └── graphs/
│       ├── secure_asts/
│       └── vulnerable_asts/
├── models/
│   └── trained_gin_model.pth
├── phases/
│   ├── phase2_gnn_ast/
│   │   ├── model_gin.py
│   │   └── run_gnn.py
│   ├── phase3_inference/
│   │   └── infer_single_contract.py
│   └── phase4_batch_logger/
│       └── batch_infer_log.py
└── README.md
```

---

## Requirements

- Python 3.11
- PyTorch
- PyTorch Geometric
- Solidity compiler (solc v0.7.0+)

Install Python dependencies:
```sh
pip install torch torch-geometric web3 solcx
```

---

## Usage




### Single Contract Inference

Infer and log the vulnerability status of a single smart contract:

```sh
python phases/phase3_inference/infer_single_contract.py \
  --input data/raw/vulnerable/<contract_name>.sol \
  --model models/trained_gin_model.pth
```

Replace `<contract_name>` with the specific smart contract filename.

---

### Train/Test Results
`baseline_model.ipynb` - consist of baseline vulnerability detector results


## Citation

If using this project for academic purposes, please cite appropriately referencing this GitHub repository.

---

## License

MIT License.
