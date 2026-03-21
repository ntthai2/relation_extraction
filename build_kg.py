"""Build and analyze a knowledge graph from extracted relation triples."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from typing import Any

import networkx as nx
from pyvis.network import Network


TYPE_COLORS = {
    "TASK": "#4A90D9",
    "METHOD": "#E67E22",
    "DATASET": "#27AE60",
}

RELATION_COLORS = {
    "USED-FOR": "#E74C3C",
    "CONJUNCTION": "#9B59B6",
    "EVALUATE-FOR": "#1ABC9C",
    "HYPONYM-OF": "#F39C12",
    "PART-OF": "#95A5A6",
    "FEATURE-OF": "#3498DB",
    "COMPARE": "#E91E63",
}

RELATION_TYPES = [
    "USED-FOR",
    "CONJUNCTION",
    "EVALUATE-FOR",
    "HYPONYM-OF",
    "PART-OF",
    "FEATURE-OF",
    "COMPARE",
]


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    for det in ["the ", "a ", "an "]:
        if text.startswith(det):
            text = text[len(det) :]
            break
    text = text.strip(".,;:!?()[]{}")
    acronym_map = {
        "llms": "large language models",
        "llm": "large language models",
        "large language models (llms)": "large language models",
        "large language models (llms": "large language models",
        "nlp": "natural language processing",
        "nlu": "natural language understanding",
        "nmt": "neural machine translation",
        "ner": "named entity recognition",
        "qa": "question answering",
        "mt": "machine translation",
        "rl": "reinforcement learning",
        "lm": "language model",
        "lms": "language models",
        "kgs": "knowledge graphs",
        "kg": "knowledge graph",
        "rag": "retrieval augmented generation",
        "sft": "supervised fine-tuning",
    }
    text = acronym_map.get(text, text)
    return text


def build_kg(
    triples_path: str = "triples.json",
    min_edge_weight: int = 1,
    min_node_count: int = 1,
) -> nx.MultiDiGraph:
    """
    Load triples, build and return a NetworkX MultiDiGraph.
    Applies normalization, deduplication, and optional filtering.
    """
    with open(triples_path, "r", encoding="utf-8") as f:
        triples = json.load(f)

    node_count: Counter[str] = Counter()
    node_type_count: dict[str, Counter[str]] = defaultdict(Counter)
    edge_agg: dict[tuple[str, str, str], dict[str, Any]] = {}

    for triple in triples:
        subject = normalize(str(triple.get("subject", "")))
        obj = normalize(str(triple.get("object", "")))
        relation = str(triple.get("relation", "")).strip()
        subject_type = str(triple.get("subject_type", "")).strip()
        object_type = str(triple.get("object_type", "")).strip()
        confidence = float(triple.get("confidence", 0.0))
        paper_id = str(triple.get("paper_id", "")).strip()

        if not subject or not obj or not relation:
            continue
        if subject == obj:
            continue

        node_count[subject] += 1
        node_count[obj] += 1
        if subject_type:
            node_type_count[subject][subject_type] += 1
        if object_type:
            node_type_count[obj][object_type] += 1

        key = (subject, obj, relation)
        if key not in edge_agg:
            edge_agg[key] = {
                "weight": 0,
                "confidence_sum": 0.0,
                "papers": set(),
            }

        edge_agg[key]["weight"] += 1
        edge_agg[key]["confidence_sum"] += confidence
        if paper_id:
            edge_agg[key]["papers"].add(paper_id)

    allowed_nodes = {node for node, count in node_count.items() if count >= min_node_count}

    G = nx.MultiDiGraph()

    for node in allowed_nodes:
        type_counter = node_type_count.get(node, Counter())
        node_type = type_counter.most_common(1)[0][0] if type_counter else "UNKNOWN"
        G.add_node(node, type=node_type, count=node_count[node])

    for (subject, obj, relation), stats in edge_agg.items():
        if subject not in allowed_nodes or obj not in allowed_nodes:
            continue
        weight = int(stats["weight"])
        if weight < min_edge_weight:
            continue

        confidence_mean = stats["confidence_sum"] / weight
        papers_list = sorted(stats["papers"])

        G.add_edge(
            subject,
            obj,
            relation=relation,
            weight=weight,
            confidence_mean=confidence_mean,
            papers=papers_list,
        )

    return G


def _print_ranked_nodes(title: str, rows: list[tuple[str, dict[str, Any], int]], top_n: int) -> list[dict[str, Any]]:
    print(f"\n{title}")
    print("rank | entity | type | value")
    print("-" * 60)
    out: list[dict[str, Any]] = []
    for rank, (node, attrs, value) in enumerate(rows[:top_n], start=1):
        node_type = attrs.get("type", "UNKNOWN")
        print(f"{rank:>4} | {node} | {node_type} | {value}")
        out.append({"rank": rank, "entity": node, "type": node_type, "value": value})
    return out


def analyze_kg(G: nx.MultiDiGraph) -> dict:
    """
    Compute and print basic graph statistics.
    Returns a dict of stats for saving.
    """
    unique_pairs = {(u, v) for u, v in G.edges()}
    undirected_graph = nx.Graph()
    undirected_graph.add_nodes_from(G.nodes(data=True))
    undirected_graph.add_edges_from(unique_pairs)

    basic_stats = {
        "total_nodes": G.number_of_nodes(),
        "total_unique_edges": len(unique_pairs),
        "total_edge_instances": G.number_of_edges(),
        "connected_components": nx.number_connected_components(undirected_graph) if G.number_of_nodes() > 0 else 0,
    }

    print("\nBasic Stats")
    for key, value in basic_stats.items():
        print(f"- {key}: {value}")

    node_type_dist: Counter[str] = Counter()
    for _, attrs in G.nodes(data=True):
        node_type_dist[str(attrs.get("type", "UNKNOWN"))] += 1

    print("\nNode Type Distribution")
    for node_type in ["TASK", "METHOD", "DATASET"]:
        print(f"- {node_type}: {node_type_dist.get(node_type, 0)}")

    relation_dist: Counter[str] = Counter()
    edge_records: list[dict[str, Any]] = []

    for u, v, attrs in G.edges(data=True):
        relation = str(attrs.get("relation", "UNKNOWN"))
        weight = int(attrs.get("weight", 1))
        relation_dist[relation] += weight
        edge_records.append(
            {
                "subject": u,
                "object": v,
                "relation": relation,
                "weight": weight,
            }
        )

    print("\nRelation Type Distribution")
    for relation, count in sorted(relation_dist.items(), key=lambda x: (-x[1], x[0])):
        print(f"- {relation}: {count}")

    degree_rows = sorted(
        ((n, G.nodes[n], int(G.degree(n))) for n in G.nodes()),
        key=lambda x: (-x[2], x[0]),
    )
    top_degree = _print_ranked_nodes("Top 20 Nodes by Degree", degree_rows, 20)

    indegree_rows = sorted(
        ((n, G.nodes[n], int(G.in_degree(n))) for n in G.nodes()),
        key=lambda x: (-x[2], x[0]),
    )
    top_indegree = _print_ranked_nodes("Top 20 Nodes by In-Degree", indegree_rows, 20)

    top_edges = sorted(edge_records, key=lambda x: (-x["weight"], x["subject"], x["relation"], x["object"]))[:10]
    print("\nTop 10 Most Frequent Edges")
    print("rank | subject -> relation -> object | weight")
    print("-" * 80)
    for rank, rec in enumerate(top_edges, start=1):
        print(f"{rank:>4} | {rec['subject']} -> {rec['relation']} -> {rec['object']} | {rec['weight']}")

    per_relation_top: dict[str, list[dict[str, Any]]] = {}
    print("\nPer Relation Top 5 Edges")
    for relation in RELATION_TYPES:
        rel_edges = [rec for rec in edge_records if rec["relation"] == relation]
        rel_edges_sorted = sorted(rel_edges, key=lambda x: (-x["weight"], x["subject"], x["object"]))[:5]
        per_relation_top[relation] = rel_edges_sorted
        print(f"\n{relation}")
        if not rel_edges_sorted:
            print("  (none)")
            continue
        for rec in rel_edges_sorted:
            print(f"  {rec['subject']} -> {rec['object']} (weight={rec['weight']})")

    stats = {
        "basic_stats": basic_stats,
        "node_type_distribution": {
            "TASK": node_type_dist.get("TASK", 0),
            "METHOD": node_type_dist.get("METHOD", 0),
            "DATASET": node_type_dist.get("DATASET", 0),
        },
        "relation_type_distribution": dict(sorted(relation_dist.items(), key=lambda x: x[0])),
        "top_degree_nodes": top_degree,
        "top_indegree_nodes": top_indegree,
        "top_edges": top_edges,
        "per_relation_top_edges": per_relation_top,
    }

    with open("kg_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("\nSaved stats to kg_stats.json")

    return stats


def export_visualization(
    G: nx.MultiDiGraph,
    output_path: str = "kg_visualization.html",
    max_nodes: int = 300,
    min_edge_weight: int = 2,
) -> None:
    """
    Export an interactive pyvis HTML visualization.
    Filters to top max_nodes nodes by degree before visualizing.
    """
    net = Network(
        height="900px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#1a1a2e",
        font_color="white",
    )
    net.barnes_hut(
        gravity=-5000,
        central_gravity=0.3,
        spring_length=100,
        spring_strength=0.05,
        damping=0.09,
    )

    ranked_nodes = sorted(G.nodes(), key=lambda n: (-int(G.degree(n)), n))
    selected_nodes = set(ranked_nodes[:max_nodes])

    if selected_nodes:
        max_degree = max(int(G.degree(n)) for n in selected_nodes)
    else:
        max_degree = 1

    for node in selected_nodes:
        attrs = G.nodes[node]
        node_type = str(attrs.get("type", "UNKNOWN"))
        degree = int(G.degree(node))
        count = int(attrs.get("count", 0))
        color = TYPE_COLORS.get(node_type, "#BDC3C7")
        size = 10 + (degree / max_degree) * 40 if max_degree > 0 else 10

        title = (
            f"Entity: {node}<br>"
            f"Type: {node_type}<br>"
            f"Appears in: {count} triples<br>"
            f"Degree: {degree}"
        )

        net.add_node(node, label=node, color=color, size=size, title=title)

    for u, v, attrs in G.edges(data=True):
        if u not in selected_nodes or v not in selected_nodes:
            continue
        weight = int(attrs.get("weight", 1))
        if weight < min_edge_weight:
            continue

        relation = str(attrs.get("relation", "UNKNOWN"))
        color = RELATION_COLORS.get(relation, "#FFFFFF")
        papers = attrs.get("papers", [])
        num_papers = len(papers) if isinstance(papers, list) else 0
        title = (
            f"{u} -> {relation} -> {v}<br>"
            f"Weight: {weight}<br>"
            f"Papers: {num_papers}"
        )

        net.add_edge(
            u,
            v,
            label=relation,
            color=color,
            width=1 + min(weight, 10),
            title=title,
            arrows="to",
        )

    net.save_graph(output_path)
    print(f"Saved visualization to {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and visualize knowledge graph from triples")
    parser.add_argument("--triples", type=str, default="triples.json")
    parser.add_argument("--output-viz", type=str, default="kg_visualization.html")
    parser.add_argument("--output-stats", type=str, default="kg_stats.json")
    parser.add_argument("--min-edge-weight", type=int, default=1)
    parser.add_argument("--min-node-count", type=int, default=1)
    parser.add_argument("--viz-max-nodes", type=int, default=300)
    parser.add_argument("--viz-min-edge-weight", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    G = build_kg(
        triples_path=args.triples,
        min_edge_weight=args.min_edge_weight,
        min_node_count=args.min_node_count,
    )

    stats = analyze_kg(G)
    if args.output_stats != "kg_stats.json":
        with open(args.output_stats, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Saved stats to {args.output_stats}")

    export_visualization(
        G,
        output_path=args.output_viz,
        max_nodes=args.viz_max_nodes,
        min_edge_weight=args.viz_min_edge_weight,
    )


if __name__ == "__main__":
    main()
