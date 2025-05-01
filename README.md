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
|__ Base_Model_Results.ipynb
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
`Base_Model_Results.ipynb` - consist of baseline vulnerability detector results


## Citation
  
  1] Y. Zhuang et al., “Smart Contract Vulnerability Detection
  Using Graph Neural Networks,” IJCAI-20.
  [2] K. Xu, W. Hu, J. Leskovec, S. Jegelka, “How Powerful
  are Graph Neural Networks?,” ICLR 2019.
  [3] “Multilayer perceptron,” https://en.
  wikipedia.org/wiki/Multilayer_
  perceptron.
  [4] J. Brownlee, “A Gentle Introduction to
  Generative Adversarial Networks,” Machine
  Learning Mastery, Jun. 2018, https:
  //machinelearningmastery.com/
  gentle-introduction-to-generative-adversarial-networks/
  [5] “Graph Convolutional Networks: Introduction
  to GNNs,” Medium (Oct. 2018), https:
  //medium.com/towards-data-science/
  graph-convolutional-networks-introduction-to-gnns-
  [6] “How to Design the Most Powerful Graph Neural
  Network,” Medium (Feb. 2020), https:
  //medium.com/towards-data-science/
  how-to-design-the-most-powerful-graph-neural-network-3d18b07a6e66
---


