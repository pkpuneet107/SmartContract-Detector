import os
import json
import networkx as nx
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_ast_json(ast_json):
    try:python phases/phase3_inference/infer_single_contract.py --input data/raw/vulnerable/yourfile.sol --model models/trained_gin_model.pth
        ast_data = json.loads(ast_json)
        if "ast" in ast_data:  # solc >=0.8 format
            return ast_data["ast"]
        elif "nodeType" in ast_data and ast_data["nodeType"] == "SourceUnit":  # solc <0.8 format
            return ast_data
        else:
            print("⚠️ Unrecognized AST format.")
            return {}
    except Exception as e:
        print("Error parsing JSON:", e)
        return {}


def build_ast_graph(ast_root):
    """Recursively build a networkx.DiGraph from an AST."""
    graph = nx.DiGraph()
    
    def add_node_recursive(node, parent_id=None):
        if not isinstance(node, dict) or "nodeType" not in node:
            return
        
        node_id = id(node)
        label = node.get("nodeType", "Unknown")
        graph.add_node(node_id, label=label)

        if parent_id is not None:
            graph.add_edge(parent_id, node_id)

        # Recursively explore children
        for key, value in node.items():
            if isinstance(value, dict):
                add_node_recursive(value, node_id)
            elif isinstance(value, list):
                for child in value:
                    add_node_recursive(child, node_id)

    add_node_recursive(ast_root)
    return graph

def save_graph(graph, name, output_dir, label, draw=False):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, name + ".gpickle")
    graph.graph["label"] = label
    with open(out_path, "wb") as f:
        pickle.dump(graph, f)

    if draw:
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(graph, seed=42)
        nx.draw(graph, pos, with_labels=True, node_size=50, font_size=6)
        labels = nx.get_node_attributes(graph, 'label')
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=6)
        plt.title(f"AST Graph: {name} (Label={label})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, name + ".png"))
        plt.close()

def process_folder(ast_dir, output_dir, label, draw_limit=5):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(ast_dir) if f.endswith(".json")]

    for i, filename in enumerate(tqdm(files, desc=f"Building graphs ({label=})")):
        full_path = os.path.join(ast_dir, filename)
        with open(full_path, "r") as f:
            raw = f.read()

        # Find first JSON object
        start_idx = raw.find("{")
        if start_idx == -1:
            print(f"⚠️ Could not find JSON root in: {filename}")
            continue

        ast_data = raw[start_idx:].strip()
        if not ast_data:
            print(f"⚠️ Skipping malformed AST file: {filename}")
            continue

        # Try parsing JSON safely
        try:
            ast_root = parse_ast_json(ast_data)
        except Exception as e:
            print(f"⚠️ Failed to parse JSON from file: {filename} — {e}")
            continue

        # If AST root isn't valid
        if not ast_root:
            print(f"⚠️ Skipping AST with no valid root node: {filename}")
            continue


        graph = build_ast_graph(ast_root)
        name = filename.replace("_ast.json", "")
        save_graph(graph, name, output_dir, label, draw=(i < draw_limit))

if __name__ == "__main__":
    process_folder("data/graphs/vulnerable_asts", "data/graphs/gnn_graphs", label=1, draw_limit=3)
    process_folder("data/graphs/secure_asts", "data/graphs/gnn_graphs", label=0, draw_limit=3)
