# SmartContract Detector

Smart contracts are pivotal in blockchain applications,
yet their security vulnerabilities have led
to significant financial losses. Traditional manual
audits and rule-based systems are resourceintensive
and error-prone, motivating the use of
advanced machine learning techniques for vulnerability
detection. Initially, we investigated a broad
approach using Graph Neural Networks (GNNs)
to classify smart contracts as secure or vulnerable,
representing code structures through abstract syntax
tree (AST)-based graphs and applying Graph
Convolutional Networks (GCN), Graph Isomorphism
Networks (GIN), and Temporal Message
Passing Networks (TMP). GIN models demonstrated
superior accuracy in binary classification,
showcasing effectiveness in capturing structural
patterns of vulnerabilities. To refine our analysis
further, we concentrated specifically on reentrancy
vulnerabilities—one of the most notorious
attack vectors in smart contracts, exemplified by
the DAO exploit. We utilized a Temporal Message
Passing Network (TMPNetwork), explicitly modeling
both control and data flow as graph edges
with temporal and semantic attributes. The TMPNetwork
iteratively updated node representations
using temporal edge sequences and employed a
gated readout mechanism for accurate classification.
Our targeted model achieved a remarkable
accuracy of 95% and an F1-score of 0.93, significantly
outperforming baseline methods. This
integrated approach demonstrates a powerful advancement
in automated smart contract security
analysis.

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
