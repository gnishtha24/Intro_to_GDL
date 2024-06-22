# Introduction to Geometric Deep Learning 

This is a short introduction to our project "Geometric Deep Learning" which is based on Lecture 1 of Stanford University's "ML with Graphs course", CS224W.

## Table of Contents
1. [Introduction](#introduction)
2. [Introduction to Graphs ML and Their Representation](#introduction-to-graphs-ml-and-their-representation)
   - [2.1 Why Graphs?](#21-why-graphs)
   - [2.2 Different Types of Graphs](#22-different-types-of-graphs)
   - [2.3 Notions of Nodes, Edges, and Attributes in Graphs](#23-notions-of-nodes-edges-and-attributes-in-graphs)
   - [2.4 Basics of Graph Visualization and Network Analysis](#24-basics-of-graph-visualization-and-network-analysis)
   - [2.5 Graph Properties: Connectivity, Density, and Centrality Measures](#25-graph-properties-connectivity-density-and-centrality-measures)
3. [Applications of Graphs in the Real World](#applications-of-graphs-in-the-real-world)
   - [3.1 Node-Level Tasks and Features](#31-node-level-tasks-and-features)
4. [Applications of Graphs in ML](#applications-of-graphs-in-ml)
   - [4.1 Node-Level ML Tasks: Protein Folding](#41-node-level-ml-tasks-protein-folding)
   - [4.2 Edge-Level of Graph ML](#42-edge-level-of-graph-ml)
     - [Recommender Systems](#recommender-systems)
     - [Drug Side Effects](#drug-side-effects)
   - [4.3 Subgraph-Level ML Tasks: Traffic Prediction](#43-subgraph-level-ml-tasks-traffic-prediction)

## 1. Introduction

This is a short introduction to our project "Geometric Deep Learning" which is based on Lecture 1 of Stanford University's "ML with Graphs course", CS224W.

## 2. Introduction to Graphs ML and Their Representation

This topic introduces the concept of graphs and their representation using nodes (vertices) and edges (links). Graphs are a fundamental data structure used to model relationships and structures in various domains.

### 2.1 Why Graphs?

1. Show complex relationships and dependencies between entities.
2. Enable network analysis to understand connectivity and community structure.
3. Facilitate the development of graph algorithms for clustering and link prediction.
4. Serve as a natural representation for node and edge-level machine learning tasks.
5. Aid in visualizing large-scale networks, helping to extract meaningful insights.

### 2.2 Different Types of Graphs

The lecture discusses a variety of graph types, including weighted, unweighted, directed, and undirected graphs. Undirected graphs lack directionally-directed edges, whereas directed graphs do. In weighted graphs, edges are given values to indicate their strength or significance.

### 2.3 Notions of Nodes, Edges, and Attributes in Graphs

While edges (links) connect nodes (vertices) and represent relationships, nodes stand in for entities or elements. Both nodes and edges may be accompanied by features or attributes, adding more data for analysis.

### 2.4 Basics of Graph Visualization and Network Analysis

Graph visualization involves representing the structure and patterns of a graph visually. It helps in exploring and understanding complex graph data. Network analysis focuses on studying the properties and behaviors of a network, such as identifying communities or detecting anomalies.

### 2.5 Graph Properties: Connectivity, Density, and Centrality Measures

- **Connectivity**: Measures determine the connectedness of a graph. Connected components are sets of nodes that can be reached from each other through a series of edges. Strongly connected components exist in directed graphs, where there is a directed path between any pair of nodes.
- **Graph Density**: Quantifies the sparsity or connectedness of a graph by comparing the number of actual edges to the maximum possible edges.
- **Centrality Measures**: Help identify important nodes in a graph. Degree centrality measures the number of edges connected to a node, while betweenness centrality quantifies the extent to which a node lies on the shortest paths between other nodes.

## 3. Applications of Graphs in the Real World

Many real-world data can be naturally represented as graphs, where nodes represent entities and links represent relationships or interactions between entities. Examples include social networks, citation networks, protein-protein interaction networks, and road networks. By learning node and link-level prediction features, we can effectively capture the underlying structure and dynamics of these graphs, enabling us to extract valuable insights and make predictions.

### 3.1 Node-Level Tasks and Features

Node-level prediction focuses on predicting properties or labels associated with individual nodes in a graph. For example, in a social network, we might be interested in predicting the occupation, age group, or political affiliation of a user based on their interactions and attributes. By learning node-level prediction features, we can classify nodes into different categories or assign numerical values to them, which can be useful for various tasks such as targeted advertising, recommendation systems, and identifying anomalies.

## 4. Applications of Graphs in ML

### 4.1 Node-Level ML Tasks: Protein Folding

Protein folding is a node-level machine learning task. It involves estimating a protein's 3D structure from its amino acid sequence. In a graph, amino acids are nodes, and machine learning techniques, such as graph neural networks (GNNs), learn patterns and features at the level of the individual amino acid to forecast folding patterns. It has implications for drug discovery and disease understanding.

### 4.2 Edge-Level of Graph ML

#### Recommender Systems

Recommender systems are one type of edge-level machine learning problem in ML with Graphs. They are designed to anticipate and recommend relevant goods to users based on their preferences and previous data. In a graph, edges reflect the connections between users and items. In order to provide customized suggestions, edge-level machine learning algorithms, including graph neural networks (GNNs), collect patterns and information from user-item interactions. Applications for recommender systems in e-commerce, content platforms, and customized marketing are numerous.

#### Drug Side Effects

Predicting and comprehending the negative effects of medications on patients is an example of an edge-level machine learning task in ML with Graphs. In a graph, the connections between medications and their adverse effects are shown as edges. To find possible adverse effects, edge-level machine learning algorithms, including graph neural networks (GNNs), learn patterns and features from drug-side effect correlations. These algorithms can help with drug development, enhance patient safety, and aid in the discovery of novel drug-target interactions by analyzing the graph structure.

### 4.3 Subgraph-Level ML Tasks: Traffic Prediction

Predicting traffic conditions and patterns at the level of subgraphs inside a road network is an example of a subgraph-level machine learning task in ML with Graphs. Localized regions or road segments within the wider network are represented by subgraphs. In order to predict traffic congestion, journey durations, or accident chances, machine learning algorithms, such as graph neural networks (GNNs), learn patterns and characteristics from the subgraph structure, historical traffic data, and numerous environmental parameters. These algorithms can help with real-time traffic management, route planning, and transportation infrastructure optimization by looking at subgraphs.
