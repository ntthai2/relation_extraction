import argparse
import json
import os
import re
from collections import Counter, defaultdict
from itertools import combinations

import networkx as nx


NOISE_ENTITIES = {
    "model",
    "models",
    "data",
    "text",
    "results",
    "method",
    "methods",
    "approach",
    "system",
    "framework",
    "technique",
    "task",
    "tasks",
    "dataset",
    "datasets",
    "network",
    "networks",
    "learning",
    "training",
    "inference",
    "evaluation",
    "performance",
    "accuracy",
    "loss",
    "paper",
    "work",
    "study",
    "experiment",
    "experiments",
    "analysis",
    "human",
    "users",
    "we",
    "our",
    "it",
    "they",
    "this",
    "these",
    "large",
    "small",
    "new",
    "good",
    "better",
    "best",
    "high",
    "low",
}

ACRONYM_MAP = {
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
    "mllms": "multimodal large language models",
    "multimodal large language models (mllms": "multimodal large language models",
    "multimodal large language models (mllms)": "multimodal large language models",
    "vlms": "vision-language models",
    "vision-language models (vlms": "vision-language models",
    "vision-language models (vlms)": "vision-language models",
    "reasoning tasks": "reasoning",
    "reasoning task": "reasoning",
    "nlp tasks": "natural language processing tasks",
    "supervised fine-tuning (sft": "supervised fine-tuning",
    "supervised fine-tuning (sft)": "supervised fine-tuning",
    "reinforcement learning (rl": "reinforcement learning",
    "reinforcement learning (rl)": "reinforcement learning",
    "reinforcement learning with verifiable rewards": "rlvr",
    "federated learning (fl": "federated learning",
    "federated learning (fl)": "federated learning",
    "graph neural networks (gnns": "graph neural networks",
    "graph neural networks (gnns)": "graph neural networks",
    "graph neural network (gnn": "graph neural networks",
    "retrieval-augmented generation (rag": "retrieval augmented generation",
    "retrieval-augmented generation (rag)": "retrieval augmented generation",
    "retrieval augmented generation (rag": "retrieval augmented generation",
    "retrieval augmented generation (rag)": "retrieval augmented generation",
    "agentic rag": "agentic retrieval augmented generation",
    "naïve rag": "naive retrieval augmented generation",
    "slms": "small language models",
    "small language models (slms": "small language models",
    "small language models (slms)": "small language models",
    "llms'": "large language models",
    "llm's": "large language models",
    "llama-2": "llama",
    "llama-3": "llama",
    "llama 3": "llama",
    "llama2": "llama",
    "llama3": "llama",
    "bert-base": "bert",
    "bert-large": "bert",
    "bert model": "bert",
    "large language model": "large language models",
    "foundation model": "foundation models",
    "language model": "language models",
    "diffusion model": "diffusion models",
    "vision language model": "vision-language models",
    "gpt-4-turbo": "gpt-4",
    "gpt-4 turbo": "gpt-4",
    "gpt4": "gpt-4",
    "gpt-3.5": "gpt-3.5-turbo",
    "gpt4o": "gpt-4o",
    "3d gaussian splatting (3dgs": "3d gaussian splatting",
    "3d gaussian splatting (3dgs)": "3d gaussian splatting",
    "vision-language-action (vla) models": "vision language action models",
    "vla models": "vision language action models",
    "deep reinforcement learning (drl": "deep reinforcement learning",
    "deep reinforcement learning (drl)": "deep reinforcement learning",
    "drl": "deep reinforcement learning",
    "partial differential equations (pdes": "partial differential equations",
    "partial differential equations (pdes)": "partial differential equations",
    "pdes": "partial differential equations",
}


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(".,;:!?()[]{}")
    for det in ["the ", "a ", "an "]:
        if text.startswith(det):
            text = text[len(det) :]
            break
    # First acronym map lookup
    text = ACRONYM_MAP.get(text, text)
    # Strip trailing parenthetical abbreviation at end
    candidate = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    candidate = re.sub(r"\s*\([^)]*$", "", candidate).strip()
    text = ACRONYM_MAP.get(candidate, candidate)
    return text


def load_triples(path: str) -> list[dict]:
    """Load and normalize triples from a JSON file."""
    with open(path, encoding="utf-8") as f:
        triples = json.load(f)
    result = []
    for t in triples:
        subj = normalize(t["subject"])
        obj = normalize(t["object"])
        if not subj or not obj or subj == obj:
            continue
        if subj in NOISE_ENTITIES or obj in NOISE_ENTITIES:
            continue
        result.append(
            {
                "subject": subj,
                "subject_type": t["subject_type"],
                "relation": t["relation"],
                "object": obj,
                "object_type": t["object_type"],
                "confidence": t.get("confidence", 1.0),
                "paper_id": t.get("paper_id", ""),
                "source_sentence": t.get("source_sentence", ""),
            }
        )
    return result


def build_graph(triples: list[dict]) -> nx.MultiDiGraph:
    """Build a NetworkX MultiDiGraph from normalized triples."""
    G = nx.MultiDiGraph()
    for t in triples:
        s, o, r = t["subject"], t["object"], t["relation"]
        if not G.has_node(s):
            G.add_node(s, type=t["subject_type"], count=0)
        if not G.has_node(o):
            G.add_node(o, type=t["object_type"], count=0)
        G.nodes[s]["count"] += 1
        G.nodes[o]["count"] += 1
        found = False
        if G.has_edge(s, o):
            for key, data in G[s][o].items():
                if data["relation"] == r:
                    data["weight"] += 1
                    data["papers"].add(t["paper_id"])
                    found = True
                    break
        if not found:
            G.add_edge(s, o, relation=r, weight=1, papers={t["paper_id"]})
    return G


def paper_count(triples: list[dict]) -> int:
    count = len(set(t["paper_id"] for t in triples if t.get("paper_id")))
    return max(count, 1)


def primary_entity_types(triples: list[dict]) -> dict[str, str]:
    counts = defaultdict(Counter)
    for t in triples:
        counts[t["subject"]][t["subject_type"]] += 1
        counts[t["object"]][t["object_type"]] += 1
    out = {}
    for entity, c in counts.items():
        out[entity] = c.most_common(1)[0][0]
    return out


def safe_name(text: str) -> str:
    n = normalize(text)
    n = re.sub(r"[^a-z0-9]+", "_", n).strip("_")
    return n or "entity"


def write_json(path: str, data: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _print_simple_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    print(title)
    if not rows:
        print("(no results)")
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, col in enumerate(row):
            widths[i] = max(widths[i], len(col))
    header_line = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))))


def analysis_gap(triples_path: str, top_n: int = 50, min_cooccurrence: int = 5):
    """
    Find entity pairs that frequently co-occur in the same abstract
    but have no extracted relation between them.

    Co-occurrence: both entities appear in triples from the same paper_id.
    Gap: no triple exists with that subject-object pair in either direction.

    Prints top_n gap candidates sorted by co-occurrence count descending.
    Saves results to json/gap_findings.json
    """
    triples = load_triples(triples_path)
    entity_types = primary_entity_types(triples)

    paper_entities = defaultdict(set)
    paper_triples = defaultdict(list)
    directed_pairs = set()

    for t in triples:
        pid = t.get("paper_id", "")
        paper_entities[pid].add(t["subject"])
        paper_entities[pid].add(t["object"])
        paper_triples[pid].append(t)
        directed_pairs.add((t["subject"], t["object"]))

    cooccur_counts = Counter()
    pair_papers = defaultdict(list)
    for pid, entities in paper_entities.items():
        entities_sorted = sorted(entities)
        for a, b in combinations(entities_sorted, 2):
            cooccur_counts[(a, b)] += 1
            pair_papers[(a, b)].append(pid)

    candidates = []
    for (a, b), cnt in cooccur_counts.items():
        if cnt < min_cooccurrence:
            continue
        if (a, b) in directed_pairs or (b, a) in directed_pairs:
            continue
        sentences = []
        seen = set()
        for pid in pair_papers[(a, b)]:
            for t in paper_triples[pid]:
                sent = (t.get("source_sentence") or "").strip()
                if not sent:
                    continue
                if (
                    t["subject"] in {a, b}
                    or t["object"] in {a, b}
                    or (a in sent.lower() and b in sent.lower())
                ) and sent not in seen:
                    seen.add(sent)
                    sentences.append(sent)
                if len(sentences) >= 3:
                    break
            if len(sentences) >= 3:
                break

        candidates.append(
            {
                "entity_a": a,
                "type_a": entity_types.get(a, "UNKNOWN"),
                "entity_b": b,
                "type_b": entity_types.get(b, "UNKNOWN"),
                "cooccurrence_count": cnt,
                "example_sentences": sentences,
            }
        )

    candidates.sort(key=lambda x: x["cooccurrence_count"], reverse=True)
    top = candidates[:top_n]

    print("=== TOP GAP CANDIDATES (co-occur together but no relation extracted) ===")
    print(
        "Rank | Entity A (type) | Entity B (type) | Co-occurrences | Example sentence"
    )
    for i, item in enumerate(top, start=1):
        example = item["example_sentences"][0] if item["example_sentences"] else ""
        print(
            f"{i:>4} | {item['entity_a']} ({item['type_a']}) | "
            f"{item['entity_b']} ({item['type_b']}) | "
            f"{item['cooccurrence_count']:<14} | {example[:120]}"
        )

    out_path = os.path.join("json", "gap_findings.json")
    write_json(out_path, top)
    print(f"Saved: {out_path}")


def analysis_trend(
    triples_path_a: str,
    triples_path_b: str,
    label_a: str = "Period A",
    label_b: str = "Period B",
    top_n: int = 30,
):
    """
    Compare two sets of triples to identify trends:
    - Which entities grew most in degree (normalized by paper count)
    - Which entities declined
    - Which relations became more/less frequent
    - New entities in B not present in A
    - Entities that disappeared from A to B

    Saves results to json/trend_analysis.json
    """
    triples_a = load_triples(triples_path_a)
    triples_b = load_triples(triples_path_b)
    G_a = build_graph(triples_a)
    G_b = build_graph(triples_b)

    papers_a = paper_count(triples_a)
    papers_b = paper_count(triples_b)

    nodes_a = set(G_a.nodes())
    nodes_b = set(G_b.nodes())

    growing = []
    declining = []

    for n in sorted(nodes_a & nodes_b):
        deg_a = G_a.degree(n)
        deg_b = G_b.degree(n)
        if max(deg_a, deg_b) < 5:
            continue
        norm_a = deg_a / papers_a
        norm_b = deg_b / papers_b
        if norm_a == 0:
            continue
        growth = (norm_b - norm_a) / norm_a
        row = {
            "entity": n,
            "degree_a": deg_a,
            "degree_b": deg_b,
            "norm_degree_a": norm_a,
            "norm_degree_b": norm_b,
            "growth": growth,
            "growth_pct": growth * 100.0,
        }
        if growth >= 0:
            growing.append(row)
        else:
            declining.append(row)

    growing.sort(key=lambda x: x["growth"], reverse=True)
    declining.sort(key=lambda x: x["growth"])

    rel_a = Counter(t["relation"] for t in triples_a)
    rel_b = Counter(t["relation"] for t in triples_b)
    rel_all = sorted(set(rel_a) | set(rel_b))
    rel_comp = []
    for r in rel_all:
        na = rel_a[r] / papers_a
        nb = rel_b[r] / papers_b
        rel_comp.append(
            {
                "relation": r,
                "count_a": rel_a[r],
                "count_b": rel_b[r],
                "norm_a": na,
                "norm_b": nb,
                "delta_norm": nb - na,
            }
        )
    rel_comp.sort(key=lambda x: abs(x["delta_norm"]), reverse=True)

    new_entities = [
        {
            "entity": n,
            "degree_b": G_b.degree(n),
            "norm_degree_b": G_b.degree(n) / papers_b,
            "type": G_b.nodes[n].get("type", "UNKNOWN"),
        }
        for n in (nodes_b - nodes_a)
    ]
    new_entities.sort(key=lambda x: x["degree_b"], reverse=True)

    disappeared = [
        {
            "entity": n,
            "degree_a": G_a.degree(n),
            "norm_degree_a": G_a.degree(n) / papers_a,
            "type": G_a.nodes[n].get("type", "UNKNOWN"),
        }
        for n in (nodes_a - nodes_b)
    ]
    disappeared.sort(key=lambda x: x["degree_a"], reverse=True)

    print(f"=== TREND ANALYSIS: {label_a} vs {label_b} ===")

    grow_rows = [
        [
            str(i),
            r["entity"],
            f"{r['degree_a']}->{r['degree_b']}",
            f"{r['growth_pct']:.1f}%",
        ]
        for i, r in enumerate(growing[:top_n], start=1)
    ]
    _print_simple_table(
        "Top growing entities",
        ["Rank", "Entity", "Degree A->B", "Growth %"],
        grow_rows,
    )

    dec_rows = [
        [
            str(i),
            r["entity"],
            f"{r['degree_a']}->{r['degree_b']}",
            f"{r['growth_pct']:.1f}%",
        ]
        for i, r in enumerate(declining[:top_n], start=1)
    ]
    _print_simple_table(
        "Top declining entities",
        ["Rank", "Entity", "Degree A->B", "Growth %"],
        dec_rows,
    )

    rel_rows = [
        [
            r["relation"],
            f"{r['count_a']} ({r['norm_a']:.3f})",
            f"{r['count_b']} ({r['norm_b']:.3f})",
            f"{r['delta_norm']:+.3f}",
        ]
        for r in rel_comp[:top_n]
    ]
    _print_simple_table(
        "Relation distribution comparison (normalized by paper count)",
        ["Relation", f"{label_a}", f"{label_b}", "Delta norm"],
        rel_rows,
    )

    new_rows = [
        [str(i), r["entity"], r["type"], str(r["degree_b"])]
        for i, r in enumerate(new_entities[:20], start=1)
    ]
    _print_simple_table(
        f"New entities in {label_b}",
        ["Rank", "Entity", "Type", "Degree"],
        new_rows,
    )

    gone_rows = [
        [str(i), r["entity"], r["type"], str(r["degree_a"])]
        for i, r in enumerate(disappeared[:20], start=1)
    ]
    _print_simple_table(
        f"Disappeared entities from {label_a} to {label_b}",
        ["Rank", "Entity", "Type", "Degree"],
        gone_rows,
    )

    out = {
        "label_a": label_a,
        "label_b": label_b,
        "paper_count_a": papers_a,
        "paper_count_b": papers_b,
        "top_growing": growing[:top_n],
        "top_declining": declining[:top_n],
        "relation_comparison": rel_comp,
        "new_entities_top20": new_entities[:20],
        "disappeared_entities_top20": disappeared[:20],
    }
    out_path = os.path.join("json", "trend_analysis.json")
    write_json(out_path, out)
    print(f"Saved: {out_path}")


def analysis_method_task(
    triples_path: str, query: str, query_type: str = "auto", top_n: int = 20
):
    """
    Given a query entity name, find:
    - If query is a TASK: all methods USED-FOR it, all datasets EVALUATE-FOR it,
      related tasks via CONJUNCTION
    - If query is a METHOD: all tasks it is USED-FOR, all datasets it is
      EVALUATE-FOR or TRAINED-WITH, related methods via CONJUNCTION or HYPONYM-OF
    - query_type: "task", "method", "dataset", or "auto" (infer from graph)

    Saves results to json/method_task_{query}.json
    """
    triples = load_triples(triples_path)
    G = build_graph(triples)
    q = normalize(query)

    if q not in G:
        print(f"Warning: query entity not found in graph: {q}")
        return

    inferred = str(G.nodes[q].get("type", "UNKNOWN")).lower()
    qtype = inferred if query_type == "auto" else query_type.lower()

    mention_papers = {
        t["paper_id"]
        for t in triples
        if (t["subject"] == q or t["object"] == q) and t.get("paper_id")
    }

    rel_map = defaultdict(lambda: defaultdict(lambda: {"weight": 0, "papers": set()}))

    for _, v, _, data in G.out_edges(q, keys=True, data=True):
        rel = data.get("relation", "UNKNOWN")
        rel_map[rel][v]["weight"] += int(data.get("weight", 1))
        rel_map[rel][v]["papers"].update(data.get("papers", set()))

    for u, _, _, data in G.in_edges(q, keys=True, data=True):
        rel = data.get("relation", "UNKNOWN")
        rel_map[rel][u]["weight"] += int(data.get("weight", 1))
        rel_map[rel][u]["papers"].update(data.get("papers", set()))

    def section(relations: set[str], allowed_types: set[str] | None, limit: int):
        acc = defaultdict(lambda: {"weight": 0, "papers": set(), "type": "UNKNOWN"})
        for rel in relations:
            for ent, stats in rel_map.get(rel, {}).items():
                etype = str(G.nodes[ent].get("type", "UNKNOWN")).upper()
                if allowed_types is not None and etype not in allowed_types:
                    continue
                acc[ent]["weight"] += stats["weight"]
                acc[ent]["papers"].update(stats["papers"])
                acc[ent]["type"] = etype

        rows = []
        for ent, stats in acc.items():
            rows.append(
                {
                    "entity": ent,
                    "type": stats["type"],
                    "weight": stats["weight"],
                    "paper_count": len([p for p in stats["papers"] if p]),
                }
            )
        rows.sort(key=lambda x: (x["weight"], x["paper_count"]), reverse=True)
        return rows[:limit]

    result_sections = {}

    print(f'=== METHOD-TASK COVERAGE: "{q}" ({qtype.upper()}) ===')
    print(f"Unique papers mentioning this entity: {len(mention_papers)}")

    if qtype == "task":
        used_for = section({"USED-FOR"}, {"METHOD"}, top_n)
        eval_for = section({"EVALUATE-FOR"}, {"DATASET"}, top_n)
        related = section({"CONJUNCTION"}, {"TASK"}, min(top_n, 10))

        print(f"\nMethods USED-FOR this task (top {top_n}):")
        for i, r in enumerate(used_for, start=1):
            print(f"  {i}. {r['entity']} (weight={r['weight']}, papers={r['paper_count']})")

        print(f"\nDatasets used to EVALUATE-FOR this task (top {top_n}):")
        for i, r in enumerate(eval_for, start=1):
            print(f"  {i}. {r['entity']} (weight={r['weight']}, papers={r['paper_count']})")

        print("\nRelated tasks via CONJUNCTION (top 10):")
        for i, r in enumerate(related, start=1):
            print(f"  {i}. {r['entity']} (weight={r['weight']})")

        result_sections = {
            "methods_used_for_task": used_for,
            "datasets_evaluate_for_task": eval_for,
            "related_tasks_conjunction": related,
        }
    elif qtype == "method":
        tasks = section({"USED-FOR"}, {"TASK"}, top_n)
        datasets = section({"EVALUATE-FOR", "TRAINED-WITH"}, {"DATASET"}, top_n)
        related = section({"CONJUNCTION", "HYPONYM-OF"}, {"METHOD"}, top_n)

        print(f"\nTasks this method is USED-FOR (top {top_n}):")
        for i, r in enumerate(tasks, start=1):
            print(f"  {i}. {r['entity']} (weight={r['weight']}, papers={r['paper_count']})")

        print(f"\nDatasets linked via EVALUATE-FOR/TRAINED-WITH (top {top_n}):")
        for i, r in enumerate(datasets, start=1):
            print(f"  {i}. {r['entity']} (weight={r['weight']}, papers={r['paper_count']})")

        print(f"\nRelated methods via CONJUNCTION/HYPONYM-OF (top {top_n}):")
        for i, r in enumerate(related, start=1):
            print(f"  {i}. {r['entity']} (weight={r['weight']}, papers={r['paper_count']})")

        result_sections = {
            "tasks_used_for": tasks,
            "datasets_eval_or_trained_with": datasets,
            "related_methods": related,
        }
    else:
        generic = {}
        for rel, ent_map in rel_map.items():
            rows = []
            for ent, stats in ent_map.items():
                rows.append(
                    {
                        "entity": ent,
                        "type": str(G.nodes[ent].get("type", "UNKNOWN")),
                        "weight": stats["weight"],
                        "paper_count": len([p for p in stats["papers"] if p]),
                    }
                )
            rows.sort(key=lambda x: (x["weight"], x["paper_count"]), reverse=True)
            generic[rel] = rows[:top_n]

        print(f"\nRelation-wise connected entities (top {top_n} each):")
        for rel in sorted(generic):
            print(f"\n{rel}:")
            for i, r in enumerate(generic[rel], start=1):
                print(
                    f"  {i}. {r['entity']} ({r['type']}, weight={r['weight']}, papers={r['paper_count']})"
                )
        result_sections = {"relation_groups": generic}

    out = {
        "query": q,
        "query_type": qtype,
        "inferred_type": inferred,
        "mention_paper_count": len(mention_papers),
        "sections": result_sections,
    }
    out_path = os.path.join("json", f"method_task_{safe_name(q)}.json")
    write_json(out_path, out)
    print(f"Saved: {out_path}")


def _build_taxonomy_payload(
    H: nx.DiGraph, root_entity: str, max_depth: int
) -> dict | None:
    root = normalize(root_entity)
    if root not in H:
        return None

    children_map = defaultdict(list)
    for child, parent, data in H.edges(data=True):
        children_map[parent].append((child, int(data.get("weight", 1))))

    for p in children_map:
        children_map[p].sort(key=lambda x: x[1], reverse=True)

    levels = defaultdict(list)
    visited = {root}
    queue = [(root, 0)]

    while queue:
        node, depth = queue.pop(0)
        if depth >= max_depth:
            continue
        for child, w in children_map.get(node, []):
            if child in visited:
                continue
            visited.add(child)
            levels[depth + 1].append({"entity": child, "weight": w})
            queue.append((child, depth + 1))

    parent_levels = defaultdict(list)
    visited_up = {root}
    queue_up = [(root, 0)]
    while queue_up:
        node, depth = queue_up.pop(0)
        if depth >= max_depth:
            continue
        for parent in H.successors(node):
            if parent in visited_up:
                continue
            visited_up.add(parent)
            w = int(H[node][parent].get("weight", 1))
            parent_levels[depth + 1].append({"entity": parent, "weight": w})
            queue_up.append((parent, depth + 1))

    return {
        "root": root,
        "max_depth": max_depth,
        "children_levels": {str(k): v for k, v in sorted(levels.items())},
        "parent_levels": {str(k): v for k, v in sorted(parent_levels.items())},
    }


def _print_taxonomy_tree(H: nx.DiGraph, root_entity: str, max_depth: int) -> None:
    root = normalize(root_entity)
    children_map = defaultdict(list)
    for child, parent, data in H.edges(data=True):
        children_map[parent].append((child, int(data.get("weight", 1))))
    for p in children_map:
        children_map[p].sort(key=lambda x: x[1], reverse=True)

    print(root)

    def dfs(node: str, depth: int, prefix: str, seen: set[str]):
        if depth >= max_depth:
            return
        children = children_map.get(node, [])
        for i, (child, w) in enumerate(children):
            if child in seen:
                continue
            last = i == len(children) - 1
            branch = "└── " if last else "├── "
            print(f"{prefix}{branch}{child} (weight={w})")
            next_prefix = prefix + ("    " if last else "│   ")
            seen.add(child)
            dfs(child, depth + 1, next_prefix, seen)

    dfs(root, 0, "", {root})


def analysis_taxonomy(
    triples_path: str,
    root_entity: str = "large language models",
    max_depth: int = 3,
    min_weight: int = 2,
):
    """
    Build a taxonomy tree using HYPONYM-OF edges.
    Starting from root_entity, traverse HYPONYM-OF edges transitively
    up to max_depth levels.

    Also builds a reverse taxonomy (what is root_entity a hyponym of).

    Saves tree to json/taxonomy_{root_entity}.json
    Prints tree as indented text.
    """
    triples = load_triples(triples_path)

    hyponym_counts = Counter()
    for t in triples:
        if t["relation"] == "HYPONYM-OF":
            hyponym_counts[(t["subject"], t["object"])] += 1

    H = nx.DiGraph()
    for (child, parent), w in hyponym_counts.items():
        if w >= min_weight:
            H.add_edge(child, parent, weight=w)

    root_norm = normalize(root_entity)
    if root_norm not in H:
        print(f"Warning: taxonomy root not found in filtered HYPONYM-OF graph: {root_norm}")
        return

    payload = _build_taxonomy_payload(H, root_norm, max_depth)
    if payload is None:
        print(f"Warning: unable to build taxonomy for root: {root_norm}")
        return

    print("=== TAXONOMY TREE ===")
    _print_taxonomy_tree(H, root_norm, max_depth)

    out_path = os.path.join("json", f"taxonomy_{safe_name(root_norm)}.json")
    write_json(out_path, payload)
    print(f"Saved: {out_path}")

    # Automatically run for top 5 most common HYPONYM-OF targets.
    target_counts = Counter()
    for (_, parent), w in hyponym_counts.items():
        target_counts[parent] += w

    print("\n=== AUTO TAXONOMY ROOTS (Top 5 HYPONYM-OF targets) ===")
    for i, (target, count) in enumerate(target_counts.most_common(5), start=1):
        print(f"{i}. {target} (incoming weight={count})")
        auto_payload = _build_taxonomy_payload(H, target, max_depth)
        if auto_payload is None:
            print(f"   skipped (not in filtered graph after min_weight={min_weight})")
            continue
        auto_path = os.path.join("json", f"taxonomy_{safe_name(target)}.json")
        write_json(auto_path, auto_payload)
        print(f"   saved: {auto_path}")


def analysis_dataset_discovery(
    triples_path: str, task_filter: str = None, top_n: int = 30
):
    """
    Discover which datasets are used for which tasks.
    Uses EVALUATE-FOR edges where subject_type=DATASET or object_type=DATASET,
    and USED-FOR edges involving DATASET nodes.

    If task_filter provided, show only datasets for that specific task.
    Otherwise show top datasets overall and their associated tasks.

    Saves to json/dataset_discovery.json
    """
    triples = load_triples(triples_path)
    filt = normalize(task_filter) if task_filter else None

    info = defaultdict(
        lambda: {
            "tasks_evaluate_for": Counter(),
            "methods": Counter(),
            "papers": set(),
            "relations": Counter(),
        }
    )

    for t in triples:
        s, o, r = t["subject"], t["object"], t["relation"]
        st, ot = t["subject_type"], t["object_type"]
        pid = t.get("paper_id", "")

        dataset_nodes = []
        if st == "DATASET":
            dataset_nodes.append((s, "subject"))
        if ot == "DATASET":
            dataset_nodes.append((o, "object"))
        if not dataset_nodes:
            continue

        for ds, side in dataset_nodes:
            if pid:
                info[ds]["papers"].add(pid)
            info[ds]["relations"][r] += 1

            other = o if side == "subject" else s
            other_type = ot if side == "subject" else st

            if r == "EVALUATE-FOR" and other_type == "TASK":
                info[ds]["tasks_evaluate_for"][other] += 1

            if r in {"USED-FOR", "TRAINED-WITH"}:
                if other_type == "METHOD":
                    info[ds]["methods"][other] += 1
                elif other_type == "TASK":
                    info[ds]["tasks_evaluate_for"][other] += 1

    rows = []
    for ds, d in info.items():
        tasks = d["tasks_evaluate_for"]
        if filt and tasks.get(filt, 0) == 0:
            continue
        top_tasks = [x for x, _ in tasks.most_common(5)]
        rows.append(
            {
                "dataset": ds,
                "paper_count": len(d["papers"]),
                "top_tasks": top_tasks,
                "top_methods": [x for x, _ in d["methods"].most_common(5)],
                "relation_counts": dict(d["relations"]),
            }
        )

    rows.sort(key=lambda x: x["paper_count"], reverse=True)
    top_rows = rows[:top_n]

    print("=== DATASET DISCOVERY ===")
    print("Rank | Dataset | Papers | Top tasks it evaluates")
    for i, r in enumerate(top_rows, start=1):
        tasks = ", ".join(r["top_tasks"]) if r["top_tasks"] else "-"
        print(f"{i:>4} | {r['dataset']:<16} | {r['paper_count']:<6} | {tasks}")

    out = {
        "task_filter": filt,
        "dataset_count": len(rows),
        "results": top_rows,
    }
    out_path = os.path.join("json", "dataset_discovery.json")
    write_json(out_path, out)
    print(f"Saved: {out_path}")


def analysis_cross_domain(triples_paths: list[str], labels: list[str], top_n: int = 20):
    """
    Compare knowledge graphs across multiple domains.

    Reports:
    - Graph size comparison (nodes, edges, density)
    - Top entities per domain (normalized by paper count)
    - Shared entities across all domains (with normalized degree in each)
    - Domain-exclusive entities (appear in one domain only)
    - Relation type distribution comparison across domains
    - Top USED-FOR pairs per domain

    Saves to json/cross_domain_comparison.json
    """
    if len(triples_paths) != len(labels):
        print("Warning: --triples-list and --labels must have the same length.")
        return

    domains = []
    for path, label in zip(triples_paths, labels):
        triples = load_triples(path)
        G = build_graph(triples)
        pcount = paper_count(triples)

        metrics = {
            "label": label,
            "path": path,
            "paper_count": pcount,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G) if G.number_of_nodes() > 1 else 0.0,
            "nodes_per_1000_papers": G.number_of_nodes() * 1000.0 / pcount,
            "edges_per_1000_papers": G.number_of_edges() * 1000.0 / pcount,
        }

        top_entities = []
        for n in G.nodes():
            deg = G.degree(n)
            top_entities.append(
                {
                    "entity": n,
                    "type": G.nodes[n].get("type", "UNKNOWN"),
                    "degree": deg,
                    "norm_degree": deg / pcount,
                }
            )
        top_entities.sort(key=lambda x: x["norm_degree"], reverse=True)

        rel_counts = Counter(t["relation"] for t in triples)
        total_rel = sum(rel_counts.values()) or 1
        rel_dist = {
            rel: {
                "count": c,
                "percent": 100.0 * c / total_rel,
            }
            for rel, c in rel_counts.items()
        }

        used_for_counts = Counter()
        for u, v, _, data in G.edges(keys=True, data=True):
            if data.get("relation") == "USED-FOR":
                used_for_counts[(u, v)] += int(data.get("weight", 1))

        top_used_for = [
            {
                "subject": s,
                "object": o,
                "weight": w,
            }
            for (s, o), w in used_for_counts.most_common(10)
        ]

        domains.append(
            {
                "label": label,
                "triples": triples,
                "graph": G,
                "metrics": metrics,
                "top_entities": top_entities[:top_n],
                "relation_distribution": rel_dist,
                "top_used_for_pairs": top_used_for,
            }
        )

    node_sets = [set(d["graph"].nodes()) for d in domains]
    shared = set.intersection(*node_sets) if node_sets else set()

    shared_rows = []
    for n in shared:
        ok = True
        norm_by_domain = {}
        for d in domains:
            deg = d["graph"].degree(n)
            if deg < 3:
                ok = False
                break
            norm_by_domain[d["label"]] = deg / d["metrics"]["paper_count"]
        if ok:
            shared_rows.append(
                {
                    "entity": n,
                    "norm_degree_by_domain": norm_by_domain,
                }
            )

    shared_rows.sort(
        key=lambda x: sum(x["norm_degree_by_domain"].values()) / len(domains),
        reverse=True,
    )

    domain_presence = Counter()
    for d in domains:
        for n in d["graph"].nodes():
            domain_presence[n] += 1

    exclusive = []
    for d in domains:
        label = d["label"]
        G = d["graph"]
        entries = []
        for n in G.nodes():
            if domain_presence[n] == 1:
                entries.append(
                    {
                        "entity": n,
                        "type": G.nodes[n].get("type", "UNKNOWN"),
                        "degree": G.degree(n),
                        "norm_degree": G.degree(n) / d["metrics"]["paper_count"],
                    }
                )
        entries.sort(key=lambda x: x["degree"], reverse=True)
        exclusive.append({"label": label, "entities": entries[:20]})

    print("=== CROSS-DOMAIN COMPARISON ===")
    size_rows = []
    for d in domains:
        m = d["metrics"]
        size_rows.append(
            [
                m["label"],
                str(m["paper_count"]),
                str(m["nodes"]),
                str(m["edges"]),
                f"{m['density']:.6f}",
                f"{m['nodes_per_1000_papers']:.2f}",
                f"{m['edges_per_1000_papers']:.2f}",
            ]
        )
    _print_simple_table(
        "Graph size comparison",
        [
            "Domain",
            "Papers",
            "Nodes",
            "Edges",
            "Density",
            "Nodes/1k papers",
            "Edges/1k papers",
        ],
        size_rows,
    )

    for d in domains:
        print(f"\nTop entities: {d['label']}")
        for i, e in enumerate(d["top_entities"], start=1):
            print(
                f"  {i}. {e['entity']} ({e['type']}, deg={e['degree']}, norm={e['norm_degree']:.3f})"
            )

    print("\nShared entities across all domains (degree >= 3 in each):")
    for i, s in enumerate(shared_rows[:top_n], start=1):
        parts = [
            f"{label}: {val:.3f}"
            for label, val in sorted(s["norm_degree_by_domain"].items())
        ]
        print(f"  {i}. {s['entity']} | " + ", ".join(parts))

    print("\nDomain-exclusive entities (top 20 by degree):")
    for ex in exclusive:
        print(f"\n{ex['label']}")
        for i, e in enumerate(ex["entities"], start=1):
            print(f"  {i}. {e['entity']} ({e['type']}, degree={e['degree']})")

    print("\nRelation type distribution by domain (%):")
    relation_union = sorted(
        set().union(*[set(d["relation_distribution"].keys()) for d in domains])
    )
    for rel in relation_union:
        parts = []
        for d in domains:
            entry = d["relation_distribution"].get(rel, {"percent": 0.0, "count": 0})
            parts.append(f"{d['label']}: {entry['percent']:.2f}% ({entry['count']})")
        print(f"  {rel}: " + " | ".join(parts))

    print("\nTop USED-FOR pairs per domain:")
    for d in domains:
        print(f"\n{d['label']}")
        for i, p in enumerate(d["top_used_for_pairs"], start=1):
            print(f"  {i}. {p['subject']} -> {p['object']} (weight={p['weight']})")

    out = {
        "graph_size": [d["metrics"] for d in domains],
        "top_entities_by_domain": {
            d["label"]: d["top_entities"] for d in domains
        },
        "shared_entities": shared_rows,
        "exclusive_entities": {e["label"]: e["entities"] for e in exclusive},
        "relation_distribution": {
            d["label"]: d["relation_distribution"] for d in domains
        },
        "top_used_for_pairs": {d["label"]: d["top_used_for_pairs"] for d in domains},
    }

    out_path = os.path.join("json", "cross_domain_comparison.json")
    write_json(out_path, out)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge graph analysis tools")
    parser.add_argument(
        "--analysis",
        required=True,
        choices=[
            "gap",
            "trend",
            "method-task",
            "taxonomy",
            "dataset-discovery",
            "cross-domain",
        ],
    )

    # Shared
    parser.add_argument("--triples", default="json/triples_cscl_2025.json")
    parser.add_argument("--top-n", type=int, default=20)

    # Gap finding
    parser.add_argument("--min-cooccurrence", type=int, default=5)

    # Trend analysis
    parser.add_argument("--triples-a", default="json/triples_cscl_2024.json")
    parser.add_argument("--triples-b", default="json/triples_cscl_2025.json")
    parser.add_argument("--label-a", default="cs.CL 2024")
    parser.add_argument("--label-b", default="cs.CL 2025")

    # Method-task coverage
    parser.add_argument("--query", default="named entity recognition")
    parser.add_argument(
        "--query-type", default="auto", choices=["auto", "task", "method", "dataset"]
    )

    # Taxonomy
    parser.add_argument("--root", default="large language models")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-weight", type=int, default=2)

    # Cross-domain
    parser.add_argument(
        "--triples-list",
        nargs="+",
        default=[
            "json/triples_cscl_2025.json",
            "json/triples_cslg.json",
            "json/triples_cscv.json",
        ],
    )
    parser.add_argument(
        "--labels", nargs="+", default=["cs.CL 2025", "cs.LG 2025", "cs.CV 2025"]
    )

    args = parser.parse_args()
    os.makedirs("json", exist_ok=True)

    if args.analysis == "gap":
        analysis_gap(args.triples, args.top_n, args.min_cooccurrence)
    elif args.analysis == "trend":
        analysis_trend(args.triples_a, args.triples_b, args.label_a, args.label_b, args.top_n)
    elif args.analysis == "method-task":
        analysis_method_task(args.triples, args.query, args.query_type, args.top_n)
    elif args.analysis == "taxonomy":
        analysis_taxonomy(args.triples, args.root, args.max_depth, args.min_weight)
    elif args.analysis == "dataset-discovery":
        analysis_dataset_discovery(args.triples, top_n=args.top_n)
    elif args.analysis == "cross-domain":
        analysis_cross_domain(args.triples_list, args.labels, args.top_n)
