import networkx as nx
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from facenet_models import FacenetModel
from image_to_rgb import image_to_rgb_array
from sklearn.metrics.pairwise import cosine_distances
import random


class Node:
    """ Describes a node in a graph, and the edges connected
        to that node."""

    def __init__(self, ID, neighbors, descriptor, truth=None, file_path=None):
        """
        Parameters
        ----------
        ID : int
            A unique identifier for this node. Should be a
            value in [0, N-1], if there are N nodes in total.

        neighbors : Sequence[int]
            The node-IDs of the neighbors of this node.

        descriptor : numpy.ndarray
            The shape-(512,) descriptor vector for the face that this node corresponds to.

        truth : Optional[str]
            If you have truth data, for checking your clustering algorithm,
            you can include the label to check your clusters at the end.
            If this node corresponds to a picture of Ryan, this truth
            value can just be "Ryan"

        file_path : Optional[str]
            The file path of the image corresponding to this node, so
            that you can sort the photos after you run your clustering
            algorithm
        """
        self.id = ID  # a unique identified for this node - this should never change

        # The node's label is initialized with the node's ID value at first,
        # this label is then updated during the whispers algorithm
        self.label = ID

        # (n1_ID, n2_ID, ...)
        # The IDs of this nodes neighbors. Empty if no neighbors
        self.neighbors = tuple(neighbors)
        self.descriptor = descriptor

        self.truth = truth
        self.file_path = file_path


def plot_graph(graph, adj):
    """ Use the package networkx to produce a diagrammatic plot of the graph, with
    the nodes in the graph colored according to their current labels.
    Note that only 20 unique colors are available for the current color map,
    so common colors across nodes may be coincidental.
    Parameters
    ----------
    graph : Tuple[Node, ...]
        The graph to plot. This is simple a tuple of the nodes in the graph.
        Each element should be an instance of the `Node`-class.

    adj : numpy.ndarray, shape=(N, N)
        The adjacency-matrix for the graph. Nonzero entries indicate
        the presence of edges.

    Returns
    -------
    Tuple[matplotlib.fig.Fig, matplotlib.axis.Axes]
        The figure and axes for the plot."""

    g = nx.Graph()
    for n, node in enumerate(graph):
        g.add_node(n)

    # construct a network-x graph from the adjacency matrix: a non-zero entry at adj[i, j]
    # indicates that an egde is present between Node-i and Node-j. Because the edges are
    # undirected, the adjacency matrix must be symmetric, thus we only look ate the triangular
    # upper-half of the entries to avoid adding redundant nodes/edges
    g.add_edges_from(zip(*np.where(np.triu(adj) > 0)))

    # we want to visualize our graph of nodes and edges; to give the graph a spatial representation,
    # we treat each node as a point in 2D space, and edges like compressed springs. We simulate
    # all of these springs decompressing (relaxing) to naturally space out the nodes of the graph
    # this will hopefully give us a sensible (x, y) for each node, so that our graph is given
    # a reasonable visual depiction
    pos = nx.spring_layout(g)

    # make a mapping that maps: node-lab -> color, for each unique label in the graph
    color = list(iter(cm.tab20b(np.linspace(0, 1, len(set(i.label for i in graph))))))
    color_map = dict(zip(sorted(set(i.label for i in graph)), color))
    colors = [color_map[i.label] for i in graph]  # the color for each node in the graph, according to the node's label

    # render the visualization of the graph, with the nodes colored based on their labels!
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(g, pos=pos, ax=ax, nodelist=range(len(graph)), node_color=colors)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=g.edges())
    return fig, ax

def nodes_and_adj(images: list, threshold = 0.3):
    #returns a list of nodes and a weighted adjacency graph
    model = FacenetModel()
    descriptors = []
    valid_paths = []
    # i = 0
    for img in images:
        image = image_to_rgb_array(img)
        boxes, probabilities, _ = model.detect(image)
        
        if boxes is None or probabilities is None:
            print(f"No face detected in {img}")
            continue
        
        confident = probabilities >= threshold
        boxes = boxes[confident]
        
        descriptor = model.compute_descriptors(image, boxes)[0]
        
        descriptors.append(descriptor)
        valid_paths.append(str(img))
        # i += 1

    descriptors = np.asarray(descriptors)
    N = len(descriptors)
    adj = np.zeros((N,N))
    
    for i in range(N):
        for j in range(i+1, N):
            w = cosine_distances(descriptors[i][None, :], descriptors[j][None, :])[0, 0]
            if w < threshold:
                adj[i][j]= 1/w**2
                adj[j][i]= 1/w**2  

    nodes = []
    for i in range(N):
        neighbors = []
        for j in range(N):
            
            neighbors.append(j) if adj[i, j] > 0 and i != j else 0   
        nodes.append(Node(i, neighbors, descriptors[i]))
    
    return nodes, adj
            


def connected_components(nodes: list):
    #returns a list of lists - each inner list contains all nodes with a common label (color)
    groups = {}

    for i in nodes:
        groups.setdefault(i.label, []).append(i)

    return list(groups.values())


def propagate_label(node, nodes, adj):
    #update
    
    if len(node.neighbors) == 0:
        return node.label
    
    count = {}
    
    for neighbor_id in node.neighbors:
        neighbor = nodes[neighbor_id]
        weight = adj[node.id, neighbor_id]

        count[neighbor.label] = (count.get(neighbor.label, 0) + weight)
            
    max_val = max(count.values())
    
    best_labels = [
        label
        for label, weight in count.items()
        if np.isclose(weight, max_val)
    ]

    node.label = random.choice(best_labels)

    return node.label

def whispers(images, iterations = 10):
    nodes, adj = nodes_and_adj(images) #images should be a list of image paths
    # x = np.arange(iterations)
    y = []
    for _ in range(iterations):
        shuffled_nodes = nodes.copy()
        random.shuffle(shuffled_nodes)

        for node in shuffled_nodes:
            propagate_label(node, nodes, adj)
        plot_graph(nodes, adj)
        components = connected_components(nodes)
        y.append(len(components))
        
    plt.plot(range(iterations), y)
    plt.xlabel("Iteration")
    plt.ylabel("Number of clusters")
    plt.show()
    
    return connected_components(nodes), nodes, adj
