import os
import json
import torch
import argparse
from torch_geometric.data import Data
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from smartsec_auditor.phases.phase2_gnn_ast.model_gin import GINNet

# === Parse AST JSON into graph ===
def parse_ast_to_graph(ast_json):
    from networkx import DiGraph
    g = DiGraph()

    def add_nodes(node, parent=None):
        if not isinstance(node, dict):
            return
        node_id = id(node)
        g.add_node(node_id, label=node.get('nodeType', 'Unknown'))
        if parent is not None:
            g.add_edge(parent, node_id)
        for key, value in node.items():
            if isinstance(value, dict):
                add_nodes(value, node_id)
            elif isinstance(value, list):
                for item in value:
                    add_nodes(item, node_id)

    add_nodes(ast_json)
    return g

# === Convert graph to PyG Data ===
def graph_to_pyg_data(g, node_type_vocab):
    node_features = []
    node_id_map = {}

    for i, (node_id, attr) in enumerate(g.nodes(data=True)):
        label = attr.get('label', 'Unknown')
        idx = node_type_vocab.get(label, 0)  # default to 0 if unseen
        oh = [0] * len(node_type_vocab)
        oh[idx] = 1
        node_features.append(oh)
        node_id_map[node_id] = i

    x = torch.tensor(node_features, dtype=torch.float)
    edges = [(node_id_map[u], node_id_map[v]) for u, v in g.edges() if u in node_id_map and v in node_id_map]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    return data

# === Main ===
def main(args):
    # Step 1: Find corresponding AST
    base = os.path.basename(args.input).replace('.sol', '_ast.json')
    if '/vulnerable/' in args.input:
        ast_path = os.path.join('data', 'graphs', 'vulnerable_asts', base)
    else:
        ast_path = os.path.join('data', 'graphs', 'secure_asts', base)

    if not os.path.isfile(ast_path):
        print(f"❌ AST file not found: {ast_path}")
        return

    with open(ast_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    json_start = content.find('{')
    if json_start == -1:
        raise ValueError("No valid JSON found in AST file!")

    content = content[json_start:]
    ast_json = json.loads(content)

    # ADD THIS
    ast_root = ast_json.get('ast', ast_json)

    # Now build graph
    g = parse_ast_to_graph(ast_root)



    # Step 2: Build graph
    g = parse_ast_to_graph(ast_root)

    # Step 3: Build node type vocab
    node_type_vocab = {
        'SourceUnit': 0, 'PragmaDirective': 1, 'ContractDefinition': 2, 'InheritanceSpecifier': 3,
        'StateVariableDeclaration': 4, 'FunctionDefinition': 5, 'ModifierInvocation': 6, 
        'Block': 7, 'ExpressionStatement': 8, 'VariableDeclarationStatement': 9, 'Return': 10, 
        'Identifier': 11, 'BinaryOperation': 12, 'Literal': 13, 'MemberAccess': 14, 'IndexAccess': 15, 
        'FunctionCall': 16, 'UnaryOperation': 17, 'ElementaryTypeNameExpression': 18, 'BooleanLiteral': 19, 
        'TupleExpression': 20, 'Assignment': 21, 'VariableDeclaration': 22, 'ElementaryTypeName': 23, 
        'ParameterList': 24, 'Parameter': 25, 'ModifierDefinition': 26, 'EventDefinition': 27, 
        'StructDefinition': 28, 'EnumDefinition': 29, 'EnumValue': 30, 'Mapping': 31, 'ArrayTypeName': 32,
        'UserDefinedTypeName': 33, 'NewExpression': 34, 'ImportDirective': 35, 'Conditional': 36,
        'ForStatement': 37, 'WhileStatement': 38, 'DoWhileStatement': 39, 'IfStatement': 40,
        'Break': 41, 'Continue': 42, 'EmitStatement': 43, 'RevertStatement': 44
    }


    data = graph_to_pyg_data(g, node_type_vocab)

    # Step 4: Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GINNet(num_node_features=len(node_type_vocab), hidden_channels=128).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Step 5: Predict
    data = data.to(device)
    out = model(data.x, data.edge_index, batch=torch.zeros(data.x.size(0), dtype=torch.long).to(device))
    pred = out.argmax(dim=1).item()
    confidence = torch.softmax(out, dim=1)[0, pred].item()

    classes = ['Secure', 'Vulnerable']
    print(f"\nPrediction: {classes[pred]} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input Solidity (.sol) file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model .pth file')
    args = parser.parse_args()

    main(args)
