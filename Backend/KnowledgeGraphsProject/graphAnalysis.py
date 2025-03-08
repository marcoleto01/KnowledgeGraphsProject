import random
import re

import numpy as np
import json
import networkx as nx
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F


def preprocess_graph_data(data):
    G = nx.Graph()
    edges = set()

    for entry in data:
        nodo1 = entry["nodo1"]["id"]
        nodo2 = entry["nodo2"]["id"]

        if nodo1 != nodo2:
            edge = tuple(sorted([nodo1, nodo2]))
            if edge not in edges:
                edges.add(edge)
                G.add_edge(nodo1, nodo2)

    return G


def getGraph():
    import json
    file_path = 'GraphAnalysis/energyReportsGraph/graph_data.json'
    with open(file_path, "r") as f:
        data = json.load(f)
    f.close()
    return preprocess_graph_data(data)


def getNodeNameMapping():
    file_path = 'GraphAnalysis/energyReportsGraph/node_id_text.json'
    with open(file_path, "r") as f:
        node_id_text = json.load(f)
    f.close()
    return node_id_text


def getLabelsMapping():
    file_path = 'GraphAnalysis/energyReportsGraph/labels_mapping.json'
    with open(file_path, "r") as f:
        labels_mapping = json.load(f)
    f.close()
    return labels_mapping


def getGraphInformation(G):
    node_name_mapping = getNodeNameMapping()

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = round(nx.density(G), 3)
    avg_degree = round(sum(dict(G.degree()).values()) / num_nodes, 3) if num_nodes > 0 else 0
    avg_clustering = round(nx.average_clustering(G), 3)

    # Assortatività
    assortativity = round(nx.degree_assortativity_coefficient(G), 3)

    # Calcolo delle tre centralità principali
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)

    # Intersezione delle centralità per trovare i nodi più influenti
    intersection = set(degree_centrality.keys()) & set(betweenness_centrality.keys()) & set(closeness_centrality.keys())
    avg_centrality = {k: round((degree_centrality[k] + betweenness_centrality[k] + closeness_centrality[k]) / 3, 3)
                      for k in intersection}

    # Selezione dei 10 nodi con centralità media più alta
    top_k_centrality = sorted(avg_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    centrality_nodes = [{"node": node_name_mapping[str(node[0])], "centrality": node[1]} for node in top_k_centrality]

    # Creazione del dizionario con tutte le informazioni
    info = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,
        "avg_degree": avg_degree,
        "avg_clustering": avg_clustering,
        "assortativity": assortativity,
        "top_k_centrality": centrality_nodes
    }

    return info


def getNodeInformation(G, nodeString):
    #Get id b string
    node_mapping = getNodeNameMapping()
    print(nodeString)
    node_id = -1
    for key in node_mapping:
        if node_mapping[key] == nodeString:
            print("Trovato")
            node_id = int(key)
            break

    if node_id == -1:
        return {"error": "Nodo non presente nel grafo"}

    node_name_mapping = getNodeNameMapping()

    degree = G.degree(node_id)

    # Calcola solo le centralità per il nodo specifico
    degree_centrality = nx.degree_centrality(G).get(node_id, 0)
    betweenness_centrality = nx.betweenness_centrality(G).get(node_id, 0)
    closeness_centrality = nx.closeness_centrality(G).get(node_id, 0)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6).get(node_id, 0)
    pagerank = nx.pagerank(G).get(node_id, 0)

    # Informazioni del nodo
    node_info = {
        "node": node_name_mapping[str(node_id)],
        "degree": degree,
        "degree_centrality": degree_centrality,
        "betweenness_centrality": betweenness_centrality,
        "closeness_centrality": closeness_centrality,
        "eigenvector_centrality": eigenvector_centrality,
        "pagerank": pagerank,
    }

    return node_info


import os


def getLouvainCommunities():
    #Search for .graphml files in the directory GraphAnalysis/energyReportsGraph/louvainCommunities
    louvainCommunities = []
    for file in os.listdir('GraphAnalysis/energyReportsGraph/louvainCommunities'):
        if file.endswith(".graphml"):
            #Split the file name by . and get the first part
            louvainCommunities.append(file)

    return louvainCommunities


def getGraphById(communityId):
    communitieNames = getLouvainCommunities()
    print(communitieNames)
    for name in communitieNames:
        id = name.split('_')[0]
        if id == communityId:
            G = nx.read_graphml('GraphAnalysis/energyReportsGraph/louvainCommunities/' + name)
            if isinstance(G, nx.MultiGraph) or isinstance(G, nx.MultiDiGraph):
                G = nx.Graph(G)
            return G


def getLouvainCommunityInfo(communityId):
    G = getGraphById(communityId)

    return getGraphInformation(G)

def getAllCommunityNodesQuery(communityId):
    G = getGraphById(communityId)
    nodes = list(G.nodes)

    # Converti la lista di ID in una stringa formattata per Cypher
    id_list = ", ".join(map(str, nodes))

    # Creazione della query Cypher
    query = f"""
    MATCH (n) WHERE ID(n) IN [{id_list}]
    OPTIONAL MATCH (n)-[r]->(m)
    WHERE ID(m) IN [{id_list}]
    RETURN n, r, m
    """

    return query


def getGraphInformationByCommunity(communityId):
    G = getGraphById(communityId)
    return getGraphInformation(G)


#Creazione custom Model

from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv


class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats):  # Fix: '__init__' instead of '_init_'
        super(GCN, self).__init__()  # Fix: Correct super constructor call
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, heads=2):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, hidden_feats, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_feats * heads, hidden_feats, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats)
        self.conv2 = SAGEConv(hidden_feats, hidden_feats)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


def create_new_custom_model(uniformWeight, hardWeight, centralityWeight):
    models = {
        "GCN": GCN,
        "GAT": GAT,
        "GraphSAGE": GraphSAGE
    }

    weights = {"uniform": float(uniformWeight), "hard": float(hardWeight), "centrality": float(centralityWeight)}

    gnn_graphs_infomap = torch.load("GraphAnalysis/energyReportsGraph/data/gnn_graphs_infomap.pt",
                                    map_location=torch.device('cpu'))
    gnn_graphs_louvain = torch.load("GraphAnalysis/energyReportsGraph/data/gnn_graphs_louvain.pt",
                                    map_location=torch.device('cpu'))

    print(f"\n====== Training  ======")
    results_infomap = train_link_prediction(
        gnn_graphs_infomap,
        models,
        neg_sampling_strategy="hybrid",
        strategy_weights=weights,
        community_type="infomap"
    )

    results_louvain = train_link_prediction(
        gnn_graphs_louvain,
        models,
        neg_sampling_strategy="hybrid",
        strategy_weights=weights,
        community_type="louvain"
    )

    #Merge the results
    results = {**results_infomap, **results_louvain}

    best_model_name, best_model_data = max(results.items(), key=lambda x: (x[1]["AUC"], x[1]["AP"]))

    #Save the best model
    torch.save(best_model_data["model"].state_dict(),
               f"GraphAnalysis/energyReportsGraph/Models/{best_model_name}"
               f"_UW_{uniformWeight.split(".")[1]}HW_{hardWeight.split(".")[1]}_CW_{centralityWeight.split(".")[1]}.pth")


def create_train_test_edges(data, test_ratio=0.1, min_edges=2, neg_sampling_strategy="uniform", strategy_weights=None):
    """
    Divide gli archi in training e test set con strategia di sampling configurabile.
    """
    device = data.edge_index.device
    num_edges = data.edge_index.shape[1]
    num_test = max(int(test_ratio * num_edges), min_edges)

    perm = torch.randperm(num_edges, device=device)
    test_edges = data.edge_index[:, perm[:num_test]]
    train_edges = data.edge_index[:, perm[num_test:]]

    neg_edges = advanced_negative_sampling(
        data,
        num_neg_samples=num_test,
        strategy=neg_sampling_strategy,
        strategy_weights=strategy_weights if neg_sampling_strategy == "hybrid" else None
    )

    neg_edges = neg_edges.to(device)

    return train_edges.long(), test_edges.long(), neg_edges.long()


def link_predictor(embeddings, edge_index):
    """Dot product tra coppie di nodi per predire il link."""
    src, dst = edge_index
    src = src.long()
    dst = dst.long()
    return (embeddings[src] * embeddings[dst]).sum(dim=1)


def train_link_prediction(graphs, models, epochs=100, lr=0.01, neg_sampling_strategy="hard", strategy_weights=None,
                          community_type=None):
    device = torch.device("cpu")
    results = {}

    for model_name, Model in models.items():
        print(f"\nTraining {model_name} with negative sampling: {neg_sampling_strategy}")

        data_example = next(iter(graphs.values()))
        model = Model(in_feats=data_example.x.shape[1], hidden_feats=32).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        auc_scores = []
        ap_scores = []

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)  # Inizializziamo come tensor
            num_graphs = 0

            curr_weights = strategy_weights if neg_sampling_strategy == "hybrid" else None

            for graph_id, data in graphs.items():
                try:
                    data = data.to(device)
                    train_edges, test_edges, neg_edges = create_train_test_edges(
                        data,
                        neg_sampling_strategy=neg_sampling_strategy,
                        strategy_weights=curr_weights
                    )

                    if test_edges.size(1) < 2 or neg_edges.size(1) < 2:
                        print(f"Skipping Graph {graph_id} - Test set too small")
                        continue

                    embeddings = model(data.x, train_edges)
                    pos_pred = link_predictor(embeddings, train_edges)
                    neg_pred = link_predictor(embeddings, neg_edges)

                    labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))]).to(device)
                    pred = torch.cat([pos_pred, neg_pred])

                    loss = F.binary_cross_entropy_with_logits(pred, labels)
                    total_loss += loss
                    num_graphs += 1

                except Exception as e:
                    print(f"Errore nell'elaborazione di Graph {graph_id}: {str(e)}")
                    continue

            if num_graphs > 0:
                total_loss /= num_graphs
                total_loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Avg Loss: {total_loss.item()}")

        # Valutazione
        model.eval()
        for graph_id, data in graphs.items():

            data = data.to(device)
            train_edges, test_edges, neg_edges = create_train_test_edges(
                data,
                neg_sampling_strategy=neg_sampling_strategy,
                strategy_weights=strategy_weights if neg_sampling_strategy == "hybrid" else None
            )

            with torch.no_grad():
                embeddings = model(data.x, train_edges)
                pos_test_pred = link_predictor(embeddings, test_edges)
                neg_test_pred = link_predictor(embeddings, neg_edges)

                test_labels = torch.cat([torch.ones(pos_test_pred.size(0)),
                                         torch.zeros(neg_test_pred.size(0))]).cpu().numpy()
                test_pred = torch.cat([pos_test_pred, neg_test_pred]).cpu().numpy()

                if len(test_labels) < 2 or len(set(test_labels)) < 2:
                    print(f"Attenzione: Test set di Graph {graph_id} non valido per la valutazione")
                    continue

                auc = roc_auc_score(test_labels, test_pred)
                ap = average_precision_score(test_labels, test_pred)

                auc_scores.append(auc)
                ap_scores.append(ap)

        # Calcoliamo le metriche
        avg_auc = np.nanmean(auc_scores) if auc_scores else np.nan
        avg_ap = np.nanmean(ap_scores) if ap_scores else np.nan

        results[model_name] = {
            "model": model,
            "AUC": avg_auc,
            "AP": avg_ap
        }

    return results


def advanced_negative_sampling(data, num_neg_samples=None, strategy="hard", strategy_weights=None):
    """
    Genera esempi negativi per link prediction con 4 strategie:
    1. Uniforme (casuale)
    2. Hard Negative (campiona nodi difficili)
    3. Basato su Centralità (sceglie nodi centrali)
    4. Hybrid (combinazione pesata delle tre strategie)
    """
    num_nodes = data.x.shape[0]
    G = nx.Graph()
    G.add_edges_from(data.edge_index.t().tolist())

    if num_neg_samples is None:
        num_neg_samples = data.edge_index.shape[1] // 2

    neg_edges = []

    if strategy == "uniform":
        neg_edges = negative_sampling(data.edge_index, num_nodes=num_nodes, num_neg_samples=num_neg_samples)

    elif strategy == "hard":
        hard_neg_edges = []
        nodes = list(G.nodes())

        while len(hard_neg_edges) < num_neg_samples:
            node = random.choice(nodes)
            neighbors = set(G.neighbors(node))

            if len(neighbors) < 2:
                continue

            for _ in range(5):
                neg_node = random.choice(nodes)
                if neg_node not in neighbors and neg_node != node:
                    hard_neg_edges.append((node, neg_node))
                    break

        neg_edges = torch.tensor(hard_neg_edges, dtype=torch.long).t().contiguous()

    elif strategy == "centrality":
        pagerank = nx.pagerank(G)
        sorted_nodes = sorted(pagerank.keys(), key=lambda x: pagerank[x], reverse=True)
        neg_edges_list = []
        nodes = list(G.nodes())

        while len(neg_edges_list) < num_neg_samples:
            high_centrality_node = random.choice(sorted_nodes[:len(sorted_nodes) // 5])
            random_node = random.choice(nodes)

            if not G.has_edge(high_centrality_node, random_node):
                neg_edges_list.append((high_centrality_node, random_node))

        neg_edges = torch.tensor(neg_edges_list, dtype=torch.long).t().contiguous()

    elif strategy == "hybrid":
        if strategy_weights is None:
            raise ValueError("Per la strategia 'hybrid', è necessario fornire 'strategy_weights'.")

        total_weight = sum(strategy_weights.values())
        strategy_weights = {k: v / total_weight for k, v in strategy_weights.items()}

        num_uniform = int(num_neg_samples * strategy_weights.get("uniform", 0))
        num_hard = int(num_neg_samples * strategy_weights.get("hard", 0))
        num_centrality = int(num_neg_samples * strategy_weights.get("centrality", 0))

        if num_uniform > 0:
            uniform_neg_edges = negative_sampling(data.edge_index, num_nodes=num_nodes,
                                                  num_neg_samples=num_uniform).t().tolist()
            neg_edges.extend(uniform_neg_edges)

        if num_hard > 0:
            hard_neg_edges = []
            nodes = list(G.nodes())

            while len(hard_neg_edges) < num_hard:
                node = random.choice(nodes)
                neighbors = set(G.neighbors(node))

                if len(neighbors) < 2:
                    continue

                for _ in range(5):
                    neg_node = random.choice(nodes)
                    if neg_node not in neighbors and neg_node != node:
                        hard_neg_edges.append((node, neg_node))
                        break

            neg_edges.extend(hard_neg_edges)

        if num_centrality > 0:
            pagerank = nx.pagerank(G)
            sorted_nodes = sorted(pagerank.keys(), key=lambda x: pagerank[x], reverse=True)
            neg_edges_list = []
            nodes = list(G.nodes())

            while len(neg_edges_list) < num_centrality:
                high_centrality_node = random.choice(sorted_nodes[:len(sorted_nodes) // 5])
                random_node = random.choice(nodes)

                if not G.has_edge(high_centrality_node, random_node):
                    neg_edges_list.append((high_centrality_node, random_node))

            neg_edges.extend(neg_edges_list)

        neg_edges = torch.tensor(neg_edges, dtype=torch.long).t().contiguous()

    else:
        raise ValueError("Strategia non valida. Scegli tra 'uniform', 'hard', 'centrality' o 'hybrid'.")

    return neg_edges


##METHODS FOR LINK PREDICTION

def load_best_model(model_path, input_dim=4,
                    hidden_dim=32):
    model_class = model_path.split("/")[-1].split("_")[0]

    print("Model Class: ", model_class)

    if model_class == "GCN":
        model_class = GCN
    elif model_class == "GAT":
        model_class = GAT
    elif model_class == "GraphSAGE":
        model_class = GraphSAGE

    device = torch.device("cpu")

    model = model_class(in_feats=input_dim, hidden_feats=hidden_dim).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find model file at {model_path}")

    model.eval()
    return model


def node_prediction(model, nodes, threshold=0.75):
    with open("GraphAnalysis/energyReportsGraph/graph_data_pred.json", "rb") as f:
        data_trasformato = json.load(f)

    data_trasformato = convert_to_pyg_data(data_trasformato)

    file_path = 'GraphAnalysis/energyReportsGraph/graph_data_pred.json'
    with open(file_path, "r") as f:
        graph_data = json.load(f)

    model_path = f"GraphAnalysis/energyReportsGraph/Models/{model}"

    best_model = load_best_model(model_path, input_dim=4)

    with open("GraphAnalysis/energyReportsGraph/labels_mapping_pred.json", "rb") as f:
        labels_data = json.load(f)

    entity_mapping, entity_category_mapping = build_entity_mapping(labels_data)

    node_to_entity, node_to_category = build_node_to_entity_mapping(graph_data, entity_mapping, entity_category_mapping)

    valid_ids, entita_to_node = get_valid_ids(graph_data)

    if len(nodes) == 2:
        test_input1 = nodes[0]
        test_input2 = nodes[1]

        results_category = predict_relationship(best_model, data_trasformato, test_input1, test_input2, labels_data,
                                                entita_to_node, node_to_category, entity_mapping, threshold)
        print("\nCollegamenti Predetti (DATE →  ORG) con Probabilità:")
        #visualize_direct_links(results_category, node_to_entity)
        return get_direct_links_dict(results_category, node_to_entity)

    if len(nodes) == 3:
        test_input1 = nodes[0]
        test_input2 = nodes[1]
        test_input3 = nodes[2]

        results_category = predict_three_relationship(best_model, data_trasformato, test_input1, test_input2,
                                                      test_input3,
                                                      labels_data, entita_to_node, node_to_category, entity_mapping,
                                                      threshold)
        print("\nCollegamenti Predetti (DATE →  ORG → GEC) con Probabilità:")
        visualize_indirect_links(results_category, node_to_entity)
        return get_indirect_links_dict(results_category, node_to_entity)

    if len(nodes) == 4:
        print("SONO QUI")
        test_input1 = nodes[0]
        test_input2 = nodes[1]
        test_input3 = nodes[2]
        test_input4 = nodes[3]

        results_category = predict_four_relationship(best_model, data_trasformato, test_input1, test_input2,
                                                     test_input3, test_input4, labels_data, entita_to_node,
                                                     node_to_category,
                                                     entity_mapping, threshold)

        print("\nCollegamenti Predetti (DATE →  ORG → GEC → RES) con Probabilità:")
        visualize_four_links(results_category, node_to_entity)
        return get_four_links_dict(results_category, node_to_entity)

    if len(nodes) == 5:
        test_input1 = nodes[0]
        test_input2 = nodes[1]
        test_input3 = nodes[2]
        test_input4 = nodes[3]
        test_input5 = nodes[4]

        results_category = predict_five_relationship(best_model, data_trasformato, test_input1, test_input2,
                                                     test_input3, test_input4, test_input5, labels_data, entita_to_node,
                                                     node_to_category, entity_mapping, threshold)
        print("\nCollegamenti Predetti (DATE →  ORG → GEC → RES → ID5) con Probabilità:")
        visualize_five_links(results_category, node_to_entity)
        return get_five_links_dict(results_category, node_to_entity)


def predict_five_links(model, data, id1, id2, id3, id4, id5, labels_data, entita_to_node, node_to_category,
                       threshold=0.85, top_k=10):
    org_ids = extract_and_normalize_ID(id1, labels_data, entita_to_node, node_to_category, id1)
    energy_price_ids = extract_and_normalize_ID(id2, labels_data, entita_to_node, node_to_category, id2)
    gec_scenarios_ids = extract_and_normalize_ID(id3, labels_data, entita_to_node, node_to_category, id3)
    res_ids = extract_and_normalize_ID(id4, labels_data, entita_to_node, node_to_category, id4)
    id5_ids = extract_and_normalize_ID(id5, labels_data, entita_to_node, node_to_category, id5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        existing_edges = set(map(tuple, data.edge_index.t().cpu().numpy()))

        # 1. Predici collegamenti tra ORG e EnergyPrice
        candidate_org_energy = [(org, energy) for org in org_ids for energy in energy_price_ids
                                if (org, energy) not in existing_edges and (energy, org) not in existing_edges]

        if not candidate_org_energy:
            print("No candidate " + id1 + " → " + id2 + " pairs found")
            return []

        candidate_org_energy = torch.tensor(candidate_org_energy, dtype=torch.long).to(device)
        pred_scores_org_energy = decode_link(z, candidate_org_energy.t())
        pred_probs_org_energy = torch.sigmoid(pred_scores_org_energy)

        # 2. Predici collegamenti tra EnergyPrice e GECscenarios
        candidate_energy_gec = [(energy, gec) for energy in energy_price_ids for gec in gec_scenarios_ids
                                if (energy, gec) not in existing_edges and (gec, energy) not in existing_edges]

        if not candidate_energy_gec:
            print("No candidate " + id2 + " → " + id3 + " pairs found")
            return []

        candidate_energy_gec = torch.tensor(candidate_energy_gec, dtype=torch.long).to(device)
        pred_scores_energy_gec = decode_link(z, candidate_energy_gec.t())
        pred_probs_energy_gec = torch.sigmoid(pred_scores_energy_gec)

        # 3. Predici collegamenti tra GECscenarios e RES
        candidate_gec_res = [(gec, res) for gec in gec_scenarios_ids for res in res_ids
                             if (gec, res) not in existing_edges and (res, gec) not in existing_edges]

        if not candidate_gec_res:
            print("No candidate " + id3 + " → " + id4 + " pairs found")
            return []

        candidate_gec_res = torch.tensor(candidate_gec_res, dtype=torch.long).to(device)
        pred_scores_gec_res = decode_link(z, candidate_gec_res.t())
        pred_probs_gec_res = torch.sigmoid(pred_scores_gec_res)

        # 4. Predici collegamenti tra RES e id5
        candidate_res_id5 = [(res, id5) for res in res_ids for id5 in id5_ids
                             if (res, id5) not in existing_edges and (id5, res) not in existing_edges]

        if not candidate_res_id5:
            print("No candidate " + id4 + " → " + id5 + " pairs found")
            return []

        candidate_res_id5 = torch.tensor(candidate_res_id5, dtype=torch.long).to(device)
        pred_scores_res_id5 = decode_link(z, candidate_res_id5.t())
        pred_probs_res_id5 = torch.sigmoid(pred_scores_res_id5)

        # 5. Filtra i collegamenti con probabilità alta
        indirect_links = []
        for i, (org, energy) in enumerate(candidate_org_energy.cpu().numpy()):
            for j, (energy2, gec) in enumerate(candidate_energy_gec.cpu().numpy()):
                if energy == energy2:
                    for k, (gec2, res) in enumerate(candidate_gec_res.cpu().numpy()):
                        if gec == gec2:
                            for l, (res2, id5) in enumerate(candidate_res_id5.cpu().numpy()):
                                if res == res2:
                                    prob = float(pred_probs_org_energy[i].cpu().item() *
                                                 pred_probs_energy_gec[j].cpu().item() *
                                                 pred_probs_gec_res[k].cpu().item() *
                                                 pred_probs_res_id5[l].cpu().item())

                                    if prob >= float(threshold):
                                        indirect_links.append(
                                            (int(org), int(energy), int(gec), int(res), int(id5), prob))

        return indirect_links


def visualize_five_links(indirect_links, node_to_entity):
    for org, energy, gec, res, id5, prob in indirect_links:
        print(f"{node_to_entity.get(org, f'Unknown_{org}')} → {node_to_entity.get(energy, f'Unknown_{energy}')} → "
              f"{node_to_entity.get(gec, f'Unknown_{gec}')} → {node_to_entity.get(res, f'Unknown_{res}')} → {node_to_entity.get(id5, f'Unknown_{id5}')} (Probabilità:{prob:.4f})")


def predict_five_relationship(model, data, input1, input2, input3, input4, input5, labels_data, entita_to_node,
                              node_to_category, entity_mapping, threshold=0.75):
    """
    Predice relazioni tra cinque input, che possono essere nodi (nomi di entità) o categorie (label).
    """
    # Controlliamo se gli input sono categorie
    is_category1 = input1 in labels_data
    is_category2 = input2 in labels_data
    is_category3 = input3 in labels_data
    is_category4 = input4 in labels_data
    is_category5 = input5 in labels_data

    # Controlliamo se gli input sono nomi di entità
    is_node1 = input1 in entity_mapping.values()
    is_node2 = input2 in entity_mapping.values()
    is_node3 = input3 in entity_mapping.values()
    is_node4 = input4 in entity_mapping.values()
    is_node5 = input5 in entity_mapping.values()

    # Caso 1: Categoria - Categoria - Categoria - Categoria - Nodo
    if is_category1 and is_category2 and is_category3 and is_category4 and is_node5:

        category5 = get_label_for_node(input5, labels_data)
        node5_id = get_node_id_from_name(input5, entity_mapping, entita_to_node)

        if node5_id is None or category5 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categiria valido.")
            return []

        # Otteniamo le predizioni tra le due categorie di appartenenza
        all_predictions = predict_five_links(model, data, input1, input2, input3, input4, category5, labels_data,
                                             entita_to_node, node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, n4, n5, prob) for (n1, n2, n3, n4, n5, prob) in all_predictions
            if (n5 == node5_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []

    # Caso 2:  Categoria - Categoria - Categoria - Nodo - Nodo
    if is_category1 and is_category2 and is_category3 and is_node4 and is_node5:
        category5 = get_label_for_node(input5, labels_data)
        node5_id = get_node_id_from_name(input5, entity_mapping, entita_to_node)

        category4 = get_label_for_node(input4, labels_data)
        node4_id = get_node_id_from_name(input4, entity_mapping, entita_to_node)

        if node5_id is None or category5 is None or node4_id is None or category4 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categiria valido.")
            return []

        # Otteniamo le predizioni tra le due categorie di appartenenza
        all_predictions = predict_five_links(model, data, input1, input2, input3, category4, category5, labels_data,
                                             entita_to_node, node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, n4, n5, prob) for (n1, n2, n3, n4, n5, prob) in all_predictions
            if (n5 == node5_id and n4 == node4_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []

    # Caso 3: Categoria - Categoria - Nodo - Nodo - Nodo
    if is_category1 and is_category2 and is_node3 and is_node4 and is_node5:

        category5 = get_label_for_node(input5, labels_data)
        node5_id = get_node_id_from_name(input5, entity_mapping, entita_to_node)

        category3 = get_label_for_node(input3, labels_data)
        node3_id = get_node_id_from_name(input3, entity_mapping, entita_to_node)

        category4 = get_label_for_node(input4, labels_data)
        node4_id = get_node_id_from_name(input4, entity_mapping, entita_to_node)

        if node3_id is None or category3 is None or node4_id is None or category4 is None or node5_id is None or category5 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categiria valido.")
            return []

        # Otteniamo le predizioni tra le due categorie di appartenenza
        all_predictions = predict_five_links(model, data, input1, input2, category3, category4, category5, labels_data,
                                             entita_to_node, node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, n4, n5, prob) for (n1, n2, n3, n4, n5, prob) in all_predictions
            if (n5 == node5_id and n4 == node4_id and n3 == node3_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []

    # Caso 4: Categoria - Nodo - Nodo - Nodo - Nodo
    if is_category1 and is_node2 and is_node3 and is_node4 and is_node5:

        category5 = get_label_for_node(input5, labels_data)
        node5_id = get_node_id_from_name(input5, entity_mapping, entita_to_node)

        category3 = get_label_for_node(input3, labels_data)
        node3_id = get_node_id_from_name(input3, entity_mapping, entita_to_node)

        category2 = get_label_for_node(input2, labels_data)
        node2_id = get_node_id_from_name(input2, entity_mapping, entita_to_node)

        category4 = get_label_for_node(input4, labels_data)
        node4_id = get_node_id_from_name(input4, entity_mapping, entita_to_node)

        if node2_id is None or category2 is None or node3_id is None or category3 is None or node4_id is None or category4 is None or node5_id is None or category5 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categiria valido.")
            return []

        # Otteniamo le predizioni tra le due categorie di appartenenza
        all_predictions = predict_five_links(model, data, input1, category2, category3, category4, category5,
                                             labels_data, entita_to_node, node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, n4, n5, prob) for (n1, n2, n3, n4, n5, prob) in all_predictions
            if (n5 == node5_id and n4 == node4_id and n3 == node3_id and n2 == node2_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []

    # Caso 5: Nodo - Categoria - Nodo - Nodo - Nodo
    if is_node1 and is_category2 and is_node3 and is_node4 and is_node5:

        category5 = get_label_for_node(input5, labels_data)
        node5_id = get_node_id_from_name(input5, entity_mapping, entita_to_node)

        category3 = get_label_for_node(input3, labels_data)
        node3_id = get_node_id_from_name(input3, entity_mapping, entita_to_node)

        category1 = get_label_for_node(input1, labels_data)
        node1_id = get_node_id_from_name(input1, entity_mapping, entita_to_node)

        category4 = get_label_for_node(input4, labels_data)
        node4_id = get_node_id_from_name(input4, entity_mapping, entita_to_node)

        if node1_id is None or category1 is None or node3_id is None or category3 is None or node4_id is None or category4 is None or node5_id is None or category5 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categiria valido.")
            return []

        # Otteniamo le predizioni tra le due categorie di appartenenza
        all_predictions = predict_five_links(model, data, category1, input2, category3, category4, category5,
                                             labels_data, entita_to_node, node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, n4, n5, prob) for (n1, n2, n3, n4, n5, prob) in all_predictions
            if (n5 == node5_id and n4 == node4_id and n3 == node3_id and n1 == node1_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []

    # Caso 6: Nodo - Nodo - Categoria- Nodo - Nodo
    if is_node1 and is_node1 and is_category3 and is_node4 and is_node5:

        category5 = get_label_for_node(input5, labels_data)
        node5_id = get_node_id_from_name(input5, entity_mapping, entita_to_node)

        category2 = get_label_for_node(input2, labels_data)
        node2_id = get_node_id_from_name(input2, entity_mapping, entita_to_node)

        category1 = get_label_for_node(input1, labels_data)
        node1_id = get_node_id_from_name(input1, entity_mapping, entita_to_node)

        category4 = get_label_for_node(input4, labels_data)
        node4_id = get_node_id_from_name(input4, entity_mapping, entita_to_node)

        if node1_id is None or category1 is None or node2_id is None or category2 is None or node4_id is None or category4 is None or node5_id is None or category5 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categiria valido.")
            return []

        # Otteniamo le predizioni tra le due categorie di appartenenza
        all_predictions = predict_five_links(model, data, category1, category2, input3, category4, category5,
                                             labels_data, entita_to_node, node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, n4, n5, prob) for (n1, n2, n3, n4, n5, prob) in all_predictions
            if (n5 == node5_id and n4 == node4_id and n2 == node2_id and n1 == node1_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []

    # Caso 7: Categoria - Categoria - Categoria - Nodo - Categoria
    if is_category1 and is_category2 and is_category3 and is_node4 and is_category5:
        category4 = get_label_for_node(input4, labels_data)
        node4_id = get_node_id_from_name(input4, entity_mapping, entita_to_node)

        if node4_id is None or category4 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categoria valido.")
            return []

        # Otteniamo le predizioni tra le categorie
        all_predictions = predict_five_links(model, data, input1, input2, input3, category4, input5, labels_data,
                                             entita_to_node, node_to_category, threshold)

        filtered_predictions = [
            (n1, n2, n3, n4, n5, prob) for (n1, n2, n3, n4, n5, prob) in all_predictions
            if (n4 == node4_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi nodi.")
            return []

    # Caso 8: Categoria - Categoria - Nodo - Categoria - Nodo
    if is_category1 and is_category2 and is_node3 and is_category4 and is_node5:
        category3 = get_label_for_node(input3, labels_data)
        node3_id = get_node_id_from_name(input3, entity_mapping, entita_to_node)

        category5 = get_label_for_node(input5, labels_data)
        node5_id = get_node_id_from_name(input5, entity_mapping, entita_to_node)

        if node3_id is None or category3 is None or node5_id is None or category5 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categoria valido.")
            return []

        all_predictions = predict_five_links(model, data, input1, input2, category3, input4, category5, labels_data,
                                             entita_to_node, node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, n4, n5, prob) for (n1, n2, n3, n4, n5, prob) in all_predictions
            if (n3 == node3_id and n5 == node5_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi nodi.")
            return []

    # Caso 9: Categoria - Nodo - Categoria - Nodo - Categoria
    if is_category1 and is_node2 and is_category3 and is_node4 and is_category5:
        category2 = get_label_for_node(input2, labels_data)
        node2_id = get_node_id_from_name(input2, entity_mapping, entita_to_node)

        category4 = get_label_for_node(input4, labels_data)
        node4_id = get_node_id_from_name(input4, entity_mapping, entita_to_node)

        if node2_id is None or category2 is None or node4_id is None or category4 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categoria valido.")
            return []

        all_predictions = predict_five_links(model, data, input1, category2, input3, category4, input5, labels_data,
                                             entita_to_node, node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, n4, n5, prob) for (n1, n2, n3, n4, n5, prob) in all_predictions
            if (n2 == node2_id and n4 == node4_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi nodi.")
            return []

    # Caso 10: Nodo - Categoria - Nodo - Categoria - Nodo
    if is_node1 and is_category2 and is_node3 and is_category4 and is_node5:
        category1 = get_label_for_node(input1, labels_data)
        node1_id = get_node_id_from_name(input1, entity_mapping, entita_to_node)

        category3 = get_label_for_node(input3, labels_data)
        node3_id = get_node_id_from_name(input3, entity_mapping, entita_to_node)

        category5 = get_label_for_node(input5, labels_data)
        node5_id = get_node_id_from_name(input5, entity_mapping, entita_to_node)

        if node1_id is None or category1 is None or node3_id is None or category3 is None or node5_id is None or category5 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categoria valido.")
            return []

        all_predictions = predict_five_links(model, data, category1, input2, category3, input4, category5, labels_data,
                                             entita_to_node, node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, n4, n5, prob) for (n1, n2, n3, n4, n5, prob) in all_predictions
            if (n1 == node1_id and n3 == node3_id and n5 == node5_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi nodi.")
            return []

    # Caso 11: Categoria - Categoria - Nodo - Nodo - Categoria
    if is_category1 and is_category2 and is_node3 and is_node4 and is_category5:
        category4 = get_label_for_node(input4, labels_data)
        node4_id = get_node_id_from_name(input4, entity_mapping, entita_to_node)

        category3 = get_label_for_node(input3, labels_data)
        node3_id = get_node_id_from_name(input3, entity_mapping, entita_to_node)

        if node3_id is None or category3 is None or node4_id is None or category4 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categoria valido.")
            return []

        all_predictions = predict_five_links(model, data, input1, input2, category3, category4, input5, labels_data,
                                             entita_to_node, node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, n4, n5, prob) for (n1, n2, n3, n4, n5, prob) in all_predictions
            if (n4 == node4_id and n3 == node3_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi nodi.")
            return []


    #Caso 12: Categoria - Categoria - Nodo - Categoria - Categoria
    if is_category1 and is_category2 and is_node3 and is_category4 and is_category5:

        category3 = get_label_for_node(input3, labels_data)
        node3_id = get_node_id_from_name(input3, entity_mapping, entita_to_node)

        if node3_id is None or category3 is None :
            print(f" Errore: uno dei nodi non ha un ID o Categoria valido.")
            return []

        all_predictions = predict_five_links(model, data, input1, input2, category3, input4, input5, labels_data,
                                             entita_to_node, node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, n4, n5, prob) for (n1, n2, n3, n4, n5, prob) in all_predictions
            if (n3 == node3_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi nodi.")
            return []

    # Caso 11: Tutti e cinque sono categorie
    if is_category1 and is_category2 and is_category3 and is_category4 and is_category5:
        return predict_five_links(model, data, input1, input2, input3, input4, input5, labels_data, entita_to_node,
                                  node_to_category, threshold)

    # Caso 12: Tutti e cinque sono nodi
    elif is_node1 and is_node2 and is_node3 and is_node4 and is_node5:
        categoria_1 = get_label_for_node(input1, labels_data)
        categoria_2 = get_label_for_node(input2, labels_data)
        categoria_3 = get_label_for_node(input3, labels_data)
        categoria_4 = get_label_for_node(input4, labels_data)
        categoria_5 = get_label_for_node(input5, labels_data)

        node1_id = get_node_id_from_name(input1, entity_mapping, entita_to_node)
        node2_id = get_node_id_from_name(input2, entity_mapping, entita_to_node)
        node3_id = get_node_id_from_name(input3, entity_mapping, entita_to_node)
        node4_id = get_node_id_from_name(input4, entity_mapping, entita_to_node)
        node5_id = get_node_id_from_name(input5, entity_mapping, entita_to_node)

        if node1_id is None or node2_id is None or node3_id is None or node4_id is None or node5_id is None:
            print("Errore: uno dei nodi non ha un ID valido.")
            return []

        all_predictions = predict_five_links(model, data, categoria_1, categoria_2, categoria_3, categoria_4,
                                             categoria_5, labels_data, entita_to_node, node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, n4, n5, prob) for (n1, n2, n3, n4, n5, prob) in all_predictions
            if (n5 == node5_id and n4 == node4_id and n3 == node3_id and n2 == node2_id and n1 == node1_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []

    else:
        print(" Errore: gli input devono essere tutti categorie o tutti nodi.")
        return []


def predict_four_links(model, data, id1, id2, id3, id4, labels_data, entita_to_node, node_to_category, threshold=0.75,
                       top_k=10):
    org_ids = extract_and_normalize_ID(id1, labels_data, entita_to_node, node_to_category, id1)
    energy_price_ids = extract_and_normalize_ID(id2, labels_data, entita_to_node, node_to_category, id2)
    gec_scenarios_ids = extract_and_normalize_ID(id3, labels_data, entita_to_node, node_to_category, id3)
    res_ids = extract_and_normalize_ID(id4, labels_data, entita_to_node, node_to_category, id4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = data.to(device)

    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)

        existing_edges = set(map(tuple, data.edge_index.t().cpu().numpy()))

        # 1. Predici collegamenti tra ORG e energyPrice
        candidate_org_energy = [
            (org, energy) for org in org_ids for energy in energy_price_ids
            if (org, energy) not in existing_edges and (energy, org) not in existing_edges
        ]

        if not candidate_org_energy:
            print("No candidate " + id1 + " → " + id2 + " pairs found")
            return []

        candidate_org_energy = torch.tensor(candidate_org_energy, dtype=torch.long).to(device)
        pred_scores_org_energy = decode_link(z, candidate_org_energy.t())
        pred_probs_org_energy = torch.sigmoid(pred_scores_org_energy)

        # 2. Predici collegamenti tra energyPrice e GECscenarios
        candidate_energy_gec = [
            (energy, gec) for energy in energy_price_ids for gec in gec_scenarios_ids
            if (energy, gec) not in existing_edges and (gec, energy) not in existing_edges
        ]

        if not candidate_energy_gec:
            print("No candidate " + id2 + " → " + id3 + " pairs found")
            return []

        candidate_energy_gec = torch.tensor(candidate_energy_gec, dtype=torch.long).to(device)
        pred_scores_energy_gec = decode_link(z, candidate_energy_gec.t())
        pred_probs_energy_gec = torch.sigmoid(pred_scores_energy_gec)

        candidate_gec_res = [
            (gec, res) for gec in gec_scenarios_ids for res in res_ids
            if (gec, res) not in existing_edges and (res, gec) not in existing_edges
        ]

        if not candidate_gec_res:
            print("No candidate " + id3 + " → " + id4 + " pairs found")
            return []

        candidate_gec_res = torch.tensor(candidate_gec_res, dtype=torch.long).to(device)
        pred_scores_gec_res = decode_link(z, candidate_gec_res.t())
        pred_probs_gec_res = torch.sigmoid(pred_scores_gec_res)

        indirect_links = []
        for i, (org, energy) in enumerate(candidate_org_energy.cpu().numpy()):
            for j, (energy2, gec) in enumerate(candidate_energy_gec.cpu().numpy()):
                if energy == energy2:
                    for k, (gec2, res) in enumerate(candidate_gec_res.cpu().numpy()):
                        if gec == gec2:
                            prob = float(pred_probs_org_energy[i].cpu().item() * pred_probs_energy_gec[j].cpu().item() *
                                         pred_probs_gec_res[k].cpu().item())

                            if prob >= float(threshold):
                                indirect_links.append((int(org), int(energy), int(gec), int(res), prob))
        return indirect_links


def predict_four_relationship(model, data, input1, input2, input3, input4, labels_data, entita_to_node,
                              node_to_category, entity_mapping, threshold=0.75):
    is_category1 = input1 in labels_data
    is_category2 = input2 in labels_data
    is_category3 = input3 in labels_data
    is_category4 = input4 in labels_data

    is_node1 = input1 in entity_mapping.values()
    is_node2 = input2 in entity_mapping.values()
    is_node3 = input3 in entity_mapping.values()
    is_node4 = input4 in entity_mapping.values()

    # Caso 1: Categoria - Categoria - Categoria - Nodo
    if is_category1 and is_category2 and is_category3 and is_node4:
        print("Caso 1")

        category4 = get_label_for_node(input4, labels_data)
        node4_id = get_node_id_from_name(input4, entity_mapping, entita_to_node)

        if node4_id is None or category4 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categiria valido.")
            return []

        print(input1, input2, input3, category4, threshold)

        # Otteniamo le predizioni tra le due categorie di appartenenza
        all_predictions = predict_four_links(model, data, input1, input2, input3, category4, labels_data,
                                             entita_to_node, node_to_category, threshold)

        filtered_predictions = [
            (n1, n2, n3, n4, prob) for (n1, n2, n3, n4, prob) in all_predictions
            if (n4 == node4_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []

    # Caso 2: Categoria - Categoria - Nodo - Nodo
    if is_category1 and is_category2 and is_node3 and is_node4:
        category3 = get_label_for_node(input3, labels_data)
        node3_id = get_node_id_from_name(input3, entity_mapping, entita_to_node)

        category4 = get_label_for_node(input4, labels_data)
        node4_id = get_node_id_from_name(input4, entity_mapping, entita_to_node)

        if node3_id is None or category3 is None or node3_id is None or category4 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categiria valido.")
            return []

        # Otteniamo le predizioni tra le due categorie di appartenenza
        all_predictions = predict_four_links(model, data, input1, input2, category3, category4, labels_data,
                                             entita_to_node, node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, n4, prob) for (n1, n2, n3, n4, prob) in all_predictions
            if (n3 == node3_id and n4 == node4_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []

    # Caso 3: Categoria - Nodo - Nodo - Nodo
    if is_category1 and is_node2 and is_node3 and is_node4:

        category2 = get_label_for_node(input2, labels_data)
        node2_id = get_node_id_from_name(input2, entity_mapping, entita_to_node)

        category3 = get_label_for_node(input3, labels_data)
        node3_id = get_node_id_from_name(input3, entity_mapping, entita_to_node)

        category4 = get_label_for_node(input4, labels_data)
        node4_id = get_node_id_from_name(input4, entity_mapping, entita_to_node)

        if node2_id is None or category2 is None or node3_id is None or category3 is None or node3_id is None or category4 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categiria valido.")
            return []

        # Otteniamo le predizioni tra le due categorie di appartenenza
        all_predictions = predict_four_links(model, data, input1, category2, category3, category4, labels_data,
                                             entita_to_node, node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, n4, prob) for (n1, n2, n3, n4, prob) in all_predictions
            if (n2 == node2_id and n3 == node3_id and n4 == node4_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []

    # Caso 4: Nodo - Categoria - Nodo - Nodo
    if is_node1 and is_category2 and is_node3 and is_node4:

        category1 = get_label_for_node(input1, labels_data)
        node1_id = get_node_id_from_name(input1, entity_mapping, entita_to_node)

        category3 = get_label_for_node(input3, labels_data)
        node3_id = get_node_id_from_name(input3, entity_mapping, entita_to_node)

        category4 = get_label_for_node(input4, labels_data)
        node4_id = get_node_id_from_name(input4, entity_mapping, entita_to_node)

        if node1_id is None or category1 is None or node3_id is None or category3 is None or node4_id is None or category4 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categoria valido.")
            return []

        # Otteniamo le predizioni tra le due categorie di appartenenza
        all_predictions = predict_four_links(model, data, category1, input2, category3, category4, labels_data,
                                             entita_to_node, node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, n4, prob) for (n1, n2, n3, n4, prob) in all_predictions
            if (n1 == node1_id and n3 == node3_id and n4 == node4_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []
    # Caso 5: Nodo - Nodo - Categoria - Nodo
    if is_node1 and is_node2 and is_category3 and is_node4:

        category1 = get_label_for_node(input1, labels_data)
        node1_id = get_node_id_from_name(input1, entity_mapping, entita_to_node)

        category2 = get_label_for_node(input2, labels_data)
        node2_id = get_node_id_from_name(input2, entity_mapping, entita_to_node)

        category4 = get_label_for_node(input4, labels_data)
        node4_id = get_node_id_from_name(input4, entity_mapping, entita_to_node)

        if node1_id is None or category1 is None or node2_id is None or category2 is None or node4_id is None or category4 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categiria valido.")
            return []

        # Otteniamo le predizioni tra le due categorie di appartenenza
        all_predictions = predict_four_links(model, data, category1, category2, input3, category4, labels_data,
                                             entita_to_node, node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, n4, prob) for (n1, n2, n3, n4, prob) in all_predictions
            if (n1 == node1_id and n2 == node2_id and n4 == node4_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []

    # Caso 6: Nodo - Nodo - Nodo - Categoria
    if is_node1 and is_node2 and is_node3 and is_category4:

        category1 = get_label_for_node(input1, labels_data)
        node1_id = get_node_id_from_name(input1, entity_mapping, entita_to_node)

        category2 = get_label_for_node(input2, labels_data)
        node2_id = get_node_id_from_name(input2, entity_mapping, entita_to_node)

        category3 = get_label_for_node(input3, labels_data)
        node3_id = get_node_id_from_name(input3, entity_mapping, entita_to_node)

        if node1_id is None or category1 is None or node2_id is None or category2 is None or node3_id is None or category3 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categiria valido.")
            return []

        # Otteniamo le predizioni tra le due categorie di appartenenza
        all_predictions = predict_four_links(model, data, category1, category2, category3, input4, labels_data,
                                             entita_to_node, node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, n4, prob) for (n1, n2, n3, n4, prob) in all_predictions
            if (n1 == node1_id and n2 == node2_id and n3 == node3_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []

    ## Caso 2: Nodo - Categoria - Nodo - Categoria
    if is_node1 and is_category2 and is_node3 and is_category4:

        category1 = get_label_for_node(input1, labels_data)
        node1_id = get_node_id_from_name(input1, entity_mapping, entita_to_node)

        category3 = get_label_for_node(input3, labels_data)
        node3_id = get_node_id_from_name(input3, entity_mapping, entita_to_node)

        if node1_id is None or category1 is None or node3_id is None or category3 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categiria valido.")
            return []

        # Otteniamo le predizioni tra le due categorie di appartenenza
        all_predictions = predict_four_links(model, data, category1, input2, category3, input4, labels_data,
                                             entita_to_node, node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, n4, prob) for (n1, n2, n3, n4, prob) in all_predictions
            if (n1 == node1_id and n3 == node3_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []





    # Caso 7: Tutte categorie
    elif is_category1 and is_category2 and is_category3 and is_category4:
        return predict_four_links(model, data, input1, input2, input3, input4, labels_data,
                                  entita_to_node, node_to_category, threshold)

    # Caso 8: Tutti nodi
    elif is_node1 and is_node2 and is_node3:
        categoria_1 = get_label_for_node(input1, labels_data)
        categoria_2 = get_label_for_node(input2, labels_data)
        categoria_3 = get_label_for_node(input3, labels_data)
        categoria_4 = get_label_for_node(input4, labels_data)

        node1_id = get_node_id_from_name(input1, entity_mapping, entita_to_node)
        node2_id = get_node_id_from_name(input2, entity_mapping, entita_to_node)
        node3_id = get_node_id_from_name(input3, entity_mapping, entita_to_node)
        node4_id = get_node_id_from_name(input4, entity_mapping, entita_to_node)

        if None in [node1_id, node2_id, node3_id, node4_id]:
            print("Errore: uno dei nodi non ha un ID valido.")
            return []

        all_predictions = predict_four_links(model, data, categoria_1, categoria_2, categoria_3, categoria_4,
                                             labels_data,
                                             entita_to_node, node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, n4, prob) for (n1, n2, n3, n4, prob) in all_predictions
            if (n1 == node1_id and n2 == node2_id and n3 == node3_id and n4 == node4_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []

    else:
        print("Errore: combinazione non valida di nodi e categorie.")
        return []


def visualize_four_links(indirect_links, node_to_entity):
    for org, energy, gec, res, prob in indirect_links:
        print(f"{node_to_entity.get(org, f'Unknown_{org}')} → {node_to_entity.get(energy, f'Unknown_{energy}')} → "
              f"{node_to_entity.get(gec, f'Unknown_{gec}')} → {node_to_entity.get(res, f'Unknown_{res}')} (Probabilità:{prob:.4f})")


def predict_three_relationship(model, data, input1, input2, input3, labels_data, entita_to_node,
                               node_to_category, entity_mapping, threshold=0.75):
    is_category1 = input1 in labels_data
    is_category2 = input2 in labels_data
    is_category3 = input3 in labels_data

    is_node1 = input1 in entity_mapping.values()
    is_node2 = input2 in entity_mapping.values()
    is_node3 = input3 in entity_mapping.values()

    # Caso 1: Categoria - Categoria - Nodo
    if is_category1 and is_category2 and is_node3:

        category3 = get_label_for_node(input3, labels_data)
        node3_id = get_node_id_from_name(input3, entity_mapping, entita_to_node)

        if node3_id is None or category3 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categiria valido.")
            return []

        # Otteniamo le predizioni tra le due categorie di appartenenza
        all_predictions = predict_indirect_links(model, data, input1, input2, category3, labels_data, entita_to_node,
                                                 node_to_category, threshold)

        filtered_predictions = [
            (n1, n2, n3, prob) for (n1, n2, n3, prob) in all_predictions
            if (n3 == node3_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []

    # Caso 2: Categoria - Nodo - Nodo
    if is_category1 and is_node2 and is_node3:

        category2 = get_label_for_node(input2, labels_data)
        node2_id = get_node_id_from_name(input2, entity_mapping, entita_to_node)

        category3 = get_label_for_node(input3, labels_data)
        node3_id = get_node_id_from_name(input3, entity_mapping, entita_to_node)

        if node2_id is None or category2 is None or node3_id is None or category3 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categiria valido.")
            return []

        # Otteniamo le predizioni tra le due categorie di appartenenza
        all_predictions = predict_indirect_links(model, data, input1, category2, category3, labels_data, entita_to_node,
                                                 node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, prob) for (n1, n2, n3, prob) in all_predictions
            if (n3 == node3_id and n2 == node2_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []

    # Caso 3: Nodo - Categoria - Nodo
    if is_node1 and is_category2 and is_node3:

        category1 = get_label_for_node(input1, labels_data)
        node1_id = get_node_id_from_name(input1, entity_mapping, entita_to_node)

        category3 = get_label_for_node(input3, labels_data)
        node3_id = get_node_id_from_name(input3, entity_mapping, entita_to_node)

        if node1_id is None or category1 is None or node3_id is None or category3 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categiria valido.")
            return []

        # Otteniamo le predizioni tra le due categorie di appartenenza
        all_predictions = predict_indirect_links(model, data, category1, input2, category3, labels_data, entita_to_node,
                                                 node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, prob) for (n1, n2, n3, prob) in all_predictions
            if (n3 == node3_id and n1 == node1_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []

    # Caso 4: Nodo - Nodo - Categoria
    if is_node1 and is_node2 and is_category3:

        category1 = get_label_for_node(input1, labels_data)
        node1_id = get_node_id_from_name(input1, entity_mapping, entita_to_node)

        category2 = get_label_for_node(input2, labels_data)
        node2_id = get_node_id_from_name(input2, entity_mapping, entita_to_node)

        if node1_id is None or category1 is None or node2_id is None or category2 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categiria valido.")
            return []

        # Otteniamo le predizioni tra le due categorie di appartenenza
        all_predictions = predict_indirect_links(model, data, category1, category2, input3, labels_data, entita_to_node,
                                                 node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, prob) for (n1, n2, n3, prob) in all_predictions
            if (n1 == node1_id and n2 == node2_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []

    # Caso 5: Nodo - Categoria - Categoria
    if is_node1 and is_category2 and is_category3:
        category1 = get_label_for_node(input1, labels_data)
        node1_id = get_node_id_from_name(input1, entity_mapping, entita_to_node)

        if node1_id is None or category1 is None:
            print(f" Errore: uno dei nodi non ha un ID o Categiria valido.")
            return []

        # Otteniamo le predizioni tra le due categorie di appartenenza
        all_predictions = predict_indirect_links(model, data, category1, input2, input3, labels_data, entita_to_node,
                                                 node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, prob) for (n1, n2, n3, prob) in all_predictions
            if (n1 == node1_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []


    # Caso 6: Tutte categorie
    elif is_category1 and is_category2 and is_category3:
        return predict_indirect_links(model, data, input1, input2, input3, labels_data, entita_to_node,
                                      node_to_category, threshold)
    # Caso 7: Tutti nodi
    elif is_node1 and is_node2 and is_node3:
        categoria_1 = get_label_for_node(input1, labels_data)
        categoria_2 = get_label_for_node(input2, labels_data)
        categoria_3 = get_label_for_node(input3, labels_data)

        node1_id = get_node_id_from_name(input1, entity_mapping, entita_to_node)
        node2_id = get_node_id_from_name(input2, entity_mapping, entita_to_node)
        node3_id = get_node_id_from_name(input3, entity_mapping, entita_to_node)

        if None in [node1_id, node2_id, node3_id]:
            print("Errore: uno dei nodi non ha un ID valido.")
            return []

        all_predictions = predict_indirect_links(model, data, categoria_1, categoria_2, categoria_3, labels_data,
                                                 entita_to_node, node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, n3, prob) for (n1, n2, n3, prob) in all_predictions
            if (n1 == node1_id and n2 == node2_id and n3 == node3_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []

    # Caso 8: Niente
    else:
        print("Errore: combinazione non valida di nodi e categorie.")
        return []


def predict_indirect_links(model, data, category1, category2, category3, labels_data, entita_to_node, node_to_category,
                           threshold=0.75):
    """
    Predice collegamenti indiretti tra tre categorie con probabilità.
    """
    category1_ids = extract_and_normalize_ID(category1, labels_data, entita_to_node, node_to_category, category1)
    category2_ids = extract_and_normalize_ID(category2, labels_data, entita_to_node, node_to_category, category2)
    category3_ids = extract_and_normalize_ID(category3, labels_data, entita_to_node, node_to_category, category3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        existing_edges = set(map(tuple, data.edge_index.t().cpu().numpy()))

        # Collegamenti tra categoria1 → categoria2
        candidate_1_2 = [
            (c1, c2) for c1 in category1_ids for c2 in category2_ids
            if (c1, c2) not in existing_edges and (c2, c1) not in existing_edges
        ]

        if not candidate_1_2:
            print(f" Nessun collegamento {category1} → {category2} trovato.")
            return []

        candidate_1_2 = torch.tensor(candidate_1_2, dtype=torch.long).to(device)
        pred_scores_1_2 = decode_link(z, candidate_1_2.t())
        pred_probs_1_2 = torch.sigmoid(pred_scores_1_2)

        # Collegamenti tra categoria2 → categoria3
        candidate_2_3 = [
            (c2, c3) for c2 in category2_ids for c3 in category3_ids
            if (c2, c3) not in existing_edges and (c3, c2) not in existing_edges
        ]

        if not candidate_2_3:
            print(f" Nessun collegamento {category2} → {category3} trovato.")
            return []

        candidate_2_3 = torch.tensor(candidate_2_3, dtype=torch.long).to(device)
        pred_scores_2_3 = decode_link(z, candidate_2_3.t())
        pred_probs_2_3 = torch.sigmoid(pred_scores_2_3)

        # Filtrare i risultati validi
        indirect_links = [
            (int(c1), int(c2), int(c3), float(pred_probs_1_2[i].cpu() * pred_probs_2_3[j].cpu()))
            for i, (c1, c2) in enumerate(candidate_1_2.cpu().numpy())
            for j, (c2_match, c3) in enumerate(candidate_2_3.cpu().numpy())
            if c2 == c2_match and float(pred_probs_1_2[i].cpu() * pred_probs_2_3[j].cpu()) > float(threshold)
        ]

        return indirect_links


def visualize_indirect_links(indirect_links, node_to_entity):
    for org, energy, gec, prob in indirect_links:
        print(
            f"{node_to_entity.get(org, f'Unknown_{org}')} → {node_to_entity.get(energy, f'Unknown_{energy}')} → {node_to_entity.get(gec, f'Unknown_{gec}')} (Probabilità: {prob:.4f})")


def predict_relationship(model, data, input1, input2, labels_data, entita_to_node, node_to_category, entity_mapping,
                         threshold=0.77):
    """
    Predice relazioni tra due input, che possono essere nodi (nomi) o categorie (label).
    """

    # Controlliamo se gli input sono categorie
    is_category1 = input1 in labels_data
    is_category2 = input2 in labels_data

    # Controlliamo se gli input sono nomi di entità
    is_node1 = input1 in entity_mapping.values()
    is_node2 = input2 in entity_mapping.values()

    # Caso 1: Entrambi sono categorie
    if is_category1 and is_category2:
        return predict_direct_links(model, data, input1, input2, labels_data, entita_to_node, node_to_category,
                                    threshold)

    # Caso 2: Entrambi sono nodi
    elif is_node1 and is_node2:
        # Troviamo la label associata a ciascun nodo
        categoria_1 = get_label_for_node(input1, labels_data)
        categoria_2 = get_label_for_node(input2, labels_data)

        # Recuperiamo gli ID dei nodi
        node1_id = get_node_id_from_name(input1, entity_mapping, entita_to_node)
        node2_id = get_node_id_from_name(input2, entity_mapping, entita_to_node)

        if node1_id is None or node2_id is None:
            print(f" Errore: uno dei nodi non ha un ID valido.")
            return []

        ## labels_data, entita_to_node, node_to_category

        # Otteniamo le predizioni tra le due categorie di appartenenza
        all_predictions = predict_direct_links(model, data, categoria_1, categoria_2, labels_data, entita_to_node,
                                               node_to_category, threshold)
        filtered_predictions = [
            (n1, n2, prob) for (n1, n2, prob) in all_predictions
            if (n1 == node1_id and n2 == node2_id) or (n1 == node2_id and n2 == node1_id)
        ]

        if filtered_predictions:
            return filtered_predictions
        else:
            print(" Nessuna connessione predetta tra questi due nodi.")
            return []

    # Caso 3: Un nodo e una categoria
    elif is_node1 and is_category2:
        node_id = get_node_id_from_name(input1, entity_mapping, entita_to_node)
        category_nodes = extract_and_normalize_ID(input2, labels_data, entita_to_node, node_to_category, input2)

        if node_id is None or not category_nodes:
            print(" Errore: impossibile trovare il nodo o i nodi della categoria.")
            return []

        return predict_direct_links_with_nodes(model, data, node_id, category_nodes, threshold)

    elif is_category1 and is_node2:
        node_id = get_node_id_from_name(input2, entity_mapping, entita_to_node)
        category_nodes = extract_and_normalize_ID(input1, labels_data, entita_to_node, node_to_category, input1)

        if node_id is None or not category_nodes:
            print(" Errore: impossibile trovare il nodo o i nodi della categoria.")
            return []

        print(f"Predizione di {input2} con tutti i nodi della categoria {input1}.")
        return predict_direct_links_with_nodes(model, data, node_id, category_nodes, threshold)

    else:
        print("Errore: almeno un input deve essere un nodo o una categoria valida.")

    return []


def predict_direct_links_with_nodes(model, data, node_id, category_nodes, threshold=0.77):
    """
    Predice collegamenti tra un nodo specifico e tutti i nodi di una categoria.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    model.eval()

    with torch.no_grad():
        z = model(data.x, data.edge_index)
        existing_edges = set(map(tuple, data.edge_index.t().cpu().numpy()))

        candidates = [
            (node_id, category_node) for category_node in category_nodes
            if (node_id, category_node) not in existing_edges and (category_node, node_id) not in existing_edges
        ]

        if not candidates:
            print("Nessun candidato trovato.")
            return []

        candidates_tensor = torch.tensor(candidates, dtype=torch.long).to(device)
        pred_scores = decode_link(z, candidates_tensor.t())
        pred_probs = torch.sigmoid(pred_scores)

        results = [
            (int(n1), int(n2), float(prob.cpu()))
            for (n1, n2), prob in zip(candidates, pred_probs) if prob >= float(threshold)
        ]
        return results


# Creazione della mappa entita_id → nome_entita e categoria
def build_entity_mapping(labels_data):
    entity_mapping = {}
    entity_category_mapping = {}

    for category, category_data in labels_data.items():
        if "labels" in category_data:
            for name, entita_id in category_data["labels"].items():
                entity_mapping[int(entita_id)] = name  # Associa entita_id a nome
                entity_category_mapping[int(entita_id)] = category  # Associa entita_id a categoria

    return entity_mapping, entity_category_mapping


# Creazione della mappa node_id → nome_entita e categoria
def build_node_to_entity_mapping(graph_data, entity_mapping, entity_category_mapping):
    node_to_entity = {}
    node_to_category = {}

    for entry in graph_data:
        nodo1 = entry["nodo1"]
        nodo2 = entry["nodo2"]

        for label in nodo1["labels"]:
            entita_id_key = next((k for k in label.keys() if "entita_id" in k), None)
            if entita_id_key:
                entita_id = label[entita_id_key]
                if entita_id in entity_mapping:
                    node_to_entity[nodo1["id"]] = entity_mapping[entita_id]
                    node_to_category[nodo1["id"]] = entity_category_mapping[entita_id]

        for label in nodo2["labels"]:
            entita_id_key = next((k for k in label.keys() if "entita_id" in k), None)
            if entita_id_key:
                entita_id = label[entita_id_key]
                if entita_id in entity_mapping:
                    node_to_entity[nodo2["id"]] = entity_mapping[entita_id]
                    node_to_category[nodo2["id"]] = entity_category_mapping[entita_id]

    return node_to_entity, node_to_category


# Funzione per ottenere gli ID validi per categoria
def get_valid_ids(graph_data):
    valid_ids = set()
    entita_to_node = {}

    for entry in graph_data:
        nodo1 = entry["nodo1"]
        nodo2 = entry["nodo2"]

        for label in nodo1["labels"]:
            entita_id_key = next((k for k in label.keys() if "entita_id" in k), None)
            if entita_id_key:
                entita_id = label[entita_id_key]
                entita_to_node[entita_id] = nodo1["id"]

        for label in nodo2["labels"]:
            entita_id_key = next((k for k in label.keys() if "entita_id" in k), None)
            if entita_id_key:
                entita_id = label[entita_id_key]
                entita_to_node[entita_id] = nodo2["id"]

        valid_ids.add(nodo1["id"])
        valid_ids.add(nodo2["id"])

    return valid_ids, entita_to_node


# Funzione per estrarre e normalizzare ID per categoria
def extract_and_normalize_ID(label_category, labels_data, entita_to_node, node_to_category, required_category):
    ids = set()
    if label_category in labels_data:
        for entita, entita_id in labels_data[label_category]['labels'].items():
            if entita_id in entita_to_node:
                node_id = entita_to_node[entita_id]
                if node_id in node_to_category and node_to_category[node_id] == required_category:
                    ids.add(node_id)
    return ids


# Dato il nome di un nome ritorna la lavbel associata
def get_label_for_node(node_name, labels_mapping):
    for label, data in labels_mapping.items():
        if node_name in data["labels"]:
            return label
    return None


# Trova il node_id dato il nome del nodo
def get_node_id_from_name(node_name, entity_mapping, entita_to_node):
    name_to_entity_id = {v: k for k, v in entity_mapping.items()}
    entita_id = name_to_entity_id.get(node_name)

    if entita_id is None:
        print(f"Errore: Nome {node_name} non trovato in entity_mapping")
        return None

    node_id = entita_to_node.get(entita_id)
    if node_id is None:
        print(f"Errore: entita_id {entita_id} non trovato in entita_to_node")
        return None

    return node_id


def decode_link(embeddings, edge_index):
    "Decodifica la probabilità di connessione tra coppie di nodi usando il dot product."
    src, dst = edge_index
    return (embeddings[src] * embeddings[dst]).sum(dim=1)  # Dot product


def predict_direct_links(model, data, id1, id2, labels_data, entita_to_node, node_to_category, threshold=0.77,
                         top_k=10):
    candidate_id1 = extract_and_normalize_ID(id1, labels_data, entita_to_node, node_to_category, id1)
    candidate_id2 = extract_and_normalize_ID(id2, labels_data, entita_to_node, node_to_category, id2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = data.to(device)

    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)

        existing_edges = set(map(tuple, data.edge_index.t().cpu().numpy()))

        candidate = [
            (categoria1, categoria2) for categoria1 in candidate_id1 for categoria2 in candidate_id2
            if (categoria1, categoria2) not in existing_edges and (categoria2, categoria1) not in existing_edges
        ]

        if not candidate:
            print("No candidate " + id1 + " " + id2 + " pairs found")
            return []

        candidate = torch.tensor(candidate, dtype=torch.long).to(device)
        pred_scores = decode_link(z, candidate.t())
        pred_probs = torch.sigmoid(pred_scores)

        # Filtra i collegamenti con probabilità alta
        indirect_links = []
        for i, (categoria1, categoria2) in enumerate(candidate.cpu().numpy()):
            prob = float(pred_probs[i].cpu())
            if prob >= float(threshold):
                indirect_links.append((int(categoria1), int(categoria2), prob))

        return indirect_links


def visualize_direct_links(indirect_links, node_to_entity):
    for categoria1, categoria2, prob in indirect_links:
        print(
            f"{node_to_entity.get(categoria1, f'Unknown_{categoria1}')} → {node_to_entity.get(categoria2, f'Unknown_{categoria2}')} (Probabilità:{prob:.4f})")


def get_four_links(indirect_links, node_to_entity):
    result = []
    for org, energy, gec, res, prob in indirect_links:
        result.append(
            f"{node_to_entity.get(org, f'Unknown_{org}')} → {node_to_entity.get(energy, f'Unknown_{energy}')} → "
            f"{node_to_entity.get(gec, f'Unknown_{gec}')} → {node_to_entity.get(res, f'Unknown_{res}')} (Probabilità:{prob:.4f})")
    return result


def get_indirect_links(indirect_links, node_to_entity):
    result = []
    for org, energy, gec, prob in indirect_links:
        result.append(
            f"{node_to_entity.get(org, f'Unknown_{org}')} → {node_to_entity.get(energy, f'Unknown_{energy}')} → {node_to_entity.get(gec, f'Unknown_{gec}')} (Probabilità: {prob:.4f})")
    return result


def get_direct_links(indirect_links, node_to_entity):
    result = []
    for categoria1, categoria2, prob in indirect_links:
        result.append(
            f"{node_to_entity.get(categoria1, f'Unknown_{categoria1}')} → {node_to_entity.get(categoria2, f'Unknown_{categoria2}')} (Probabilità:{prob:.4f})")
    return result


def get_five_links(indirect_links, node_to_entity):
    result = []
    for org, energy, gec, res, id5, prob in indirect_links:
        result.append(
            f"{node_to_entity.get(org, f'Unknown_{org}')} → {node_to_entity.get(energy, f'Unknown_{energy}')} → "
            f"{node_to_entity.get(gec, f'Unknown_{gec}')} → {node_to_entity.get(res, f'Unknown_{res}')} → "
            f"{node_to_entity.get(id5, f'Unknown_{id5}')} (Probabilità:{prob:.4f})")
    return result


#Creation of dictionary for the link prediction results
def get_four_links_dict(indirect_links, node_to_entity):
    result = []
    for org, energy, gec, res, prob in indirect_links:
        result.append({
            "relation": f"{node_to_entity.get(org, f'Unknown_{org}')} → {node_to_entity.get(energy, f'Unknown_{energy}')} → "
                        f"{node_to_entity.get(gec, f'Unknown_{gec}')} → {node_to_entity.get(res, f'Unknown_{res}')}",
            "probability": round(prob, 4)
        })
    return sorted(result, key=lambda x: x["probability"], reverse=True)


def get_indirect_links_dict(indirect_links, node_to_entity):
    result = []
    for org, energy, gec, prob in indirect_links:
        result.append({
            "relation": f"{node_to_entity.get(org, f'Unknown_{org}')} → {node_to_entity.get(energy, f'Unknown_{energy}')} → {node_to_entity.get(gec, f'Unknown_{gec}')} ",
            "probability": round(prob, 4)
        })
    return sorted(result, key=lambda x: x["probability"], reverse=True)


def get_direct_links_dict(indirect_links, node_to_entity):
    result = []
    for categoria1, categoria2, prob in indirect_links:
        result.append({
            "relation": f"{node_to_entity.get(categoria1, f'Unknown_{categoria1}')} → {node_to_entity.get(categoria2, f'Unknown_{categoria2}')}",
            "probability": round(prob, 4)
        })
    return sorted(result, key=lambda x: x["probability"], reverse=True)


def get_five_links_dict(indirect_links, node_to_entity):
    result = []
    for org, energy, gec, res, id5, prob in indirect_links:
        result.append({
            "relation": f"{node_to_entity.get(org, f'Unknown_{org}')} → {node_to_entity.get(energy, f'Unknown_{energy}')} → "
                        f"{node_to_entity.get(gec, f'Unknown_{gec}')} → {node_to_entity.get(res, f'Unknown_{res}')} → "
                        f"{node_to_entity.get(id5, f'Unknown_{id5}')}",
            "probability": round(prob, 4)
        })
    return sorted(result, key=lambda x: x["probability"], reverse=True)


def getAllPossibleLinkQueryChoose():
    possibleChoose = []
    entity_families = []
    date_pattern = r'\b\d{4}\b'
    labels_mapping = getLabelsMapping()
    for label_fam in labels_mapping:
        if label_fam not in ["TIME", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "PERCENT"]:
            entity_families.append(label_fam)
            possibleChoose.append(label_fam)
            for label in labels_mapping[label_fam]['labels']:
                if label_fam == 'DATE':
                    if re.match(date_pattern, label):
                        possibleChoose.append(label)
                else:
                    possibleChoose.append(label)

    return possibleChoose, entity_families


from torch_geometric.data import Data


def convert_to_pyg_data(graph_data):
    edge_index = []
    node_features = {}

    # Track all unique node IDs
    all_nodes = set()

    for entry in graph_data:
        nodo1_id = entry["nodo1"]["id"]
        nodo2_id = entry["nodo2"]["id"]
        edge_index.append([nodo1_id, nodo2_id])

        all_nodes.add(nodo1_id)
        all_nodes.add(nodo2_id)

        if "features" in entry["nodo1"]:
            node_features[nodo1_id] = entry["nodo1"]["features"]
        if "features" in entry["nodo2"]:
            node_features[nodo2_id] = entry["nodo2"]["features"]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    num_nodes = max(all_nodes) + 1
    feature_dim = 4

    x = torch.zeros((num_nodes, feature_dim), dtype=torch.float)

    if node_features:
        feature_dim = len(next(iter(node_features.values())))
        x = torch.zeros((num_nodes, feature_dim), dtype=torch.float)
        for node_id, features in node_features.items():
            x[node_id] = torch.tensor(features, dtype=torch.float)
    else:
        for node_id in all_nodes:
            x[node_id] = torch.rand(feature_dim)

    return Data(x=x, edge_index=edge_index)


#METHODS FOR COMMUNITY REPORT

def find_and_remove_paths(G, top_nodes, leaf_nodes):
    results = {top: {'edges': [], 'paths_to_leaves': []} for top in top_nodes}
    results['connected_top_nodes'] = []

    # Fase 1 : estrarre gli archi tra i top_nodes e se stessi
    for top in top_nodes:
        results[top]['edges'] = [(u, v) for u, v in G.edges if (u == top and v == top) or (v == top and u == top)]

    # **Fase 2: Collegare i top_nodes tra loro**
    while True:
        found = False
        for i, top1 in enumerate(top_nodes):
            for j, top2 in enumerate(top_nodes):
                if i >= j:
                    continue
                try:
                    path = nx.shortest_path(G, source=top1, target=top2)
                    results['connected_top_nodes'].append(path)
                    found = True
                    G.remove_edges_from([(path[i], path[i + 1]) for i in range(len(path) - 1)])
                except nx.NetworkXNoPath:
                    continue
        if not found:
            break

    return results


def find_leaf_nodes(G):
    if G.is_directed():
        return [node for node in G.nodes if G.out_degree(node) == 0]
    else:
        return [node for node in G.nodes if G.degree(node) == 1]


def extract_subgraph_info_from_path(data, connected_top_nodes_paths):
    visited_entry = []
    # Converti `subgraph_edges` in set per ricerca veloce
    edge_id_for_path = []

    for path in connected_top_nodes_paths:
        edges = []
        for i in range(len(path) - 1):
            edges.append((path[i], path[i + 1]))
        id_path = []
        for edge in edges:
            for entry in data:
                nodo1_id = str(entry["nodo1"]["id"])
                nodo2_id = str(entry["nodo2"]["id"])

                if (edge == (nodo1_id, nodo2_id) or edge == (nodo2_id, nodo1_id)) and entry not in visited_entry:
                    id_path.append(entry["arco"]["id"])
                    visited_entry.append(entry)
                    break
        if len(id_path) == len(edges):
            edge_id_for_path.append(id_path)

    return edge_id_for_path


def get_relationship_by_id(session, rel_id):
    query = """
    MATCH ()-[r]->()
    WHERE id(r) = $relId
    RETURN {
        id: id(r),
        type: type(r),
        properties: properties(r),
        startNode: id(startNode(r)),
        endNode: id(endNode(r))
    } AS relationshipInfo
    """

    result = session.run(query, relId=rel_id)
    record = result.single()
    return record["relationshipInfo"] if record else None


def get_text_by_edge_ids(session, edge_ids):
    full_texts = []

    for result in edge_ids:
        text = ""
        for edge_id in result:
            id_dict = get_relationship_by_id(session, edge_id)
            file_name = id_dict["properties"]["documentName"]
            sentence_index = id_dict["properties"]["sentencesIndex"]
            try:
                path = os.path.join(os.path.dirname(__file__), "..", "newDocFiles", f"{file_name}.json")
                with open(path, "r") as file:
                    doc = json.load(file)
            except FileNotFoundError:
                print(f"File {file_name} not found.")
                break
            text += doc[str(sentence_index)] + " "
        full_texts.append(text)

    return full_texts


def getCommunityReport(graphId, session):
    G = getGraphById(graphId)
    print(f"Graph Id: {graphId}")

    #Open json for the data
    data = json.load(open(f"GraphAnalysis/energyReportsGraph/graph_data.json", "r"))

    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)

    importance_scores = {node: (degree_centrality[node] + betweenness_centrality[node] + closeness_centrality[node])
                         for node in G.nodes}

    main_topics = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

    top_nodes = [node for node, score in main_topics if score > 0.8]

    print(top_nodes)

    G_copy = G.copy()
    leaf_nodes = find_leaf_nodes(G_copy)

    results = find_and_remove_paths(G_copy, top_nodes, leaf_nodes)

    connected_top_nodes_paths = results['connected_top_nodes']
    connected_top_nodes_paths = sorted(connected_top_nodes_paths, key=lambda x: len(x), reverse=True)

    # Remove all the paths from the graph that have a number of top_node less than len(top_nodes)/2
    new_connected_top_nodes_paths = []
    for path in connected_top_nodes_paths:
        n_top_node = 0
        for node in path:
            if node in top_nodes:
                n_top_node += 1

        if n_top_node >= 2:
            new_connected_top_nodes_paths.append(path)

    print(connected_top_nodes_paths)

    edge_ids = extract_subgraph_info_from_path(data, connected_top_nodes_paths)

    print(edge_ids)
    texts = get_text_by_edge_ids(session, edge_ids)

    return texts
