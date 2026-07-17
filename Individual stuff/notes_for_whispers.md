Define a Node class to store the information for a picture of a single individual from your pile of pictures (see below)

Create a function that takes in a list of image-paths, and returns a list of nodes and an adjacency graph that describes the weighted connections between your nodes. This list of nodes and adjacency matrix, together, represent your graph.

You will need to estimate a maximum cosine-distance threshold for distinguishing nodes that share an edge (i.e. if 
 then node-i and node-j will share an edge). Note that this max cosine-distance threshold is also needed for the face-recognition database, so make sure that both legs of the project are collaborating to estimate this value and that you use the same value.

For your adjacency matrix, it is recommended that use weighted edges rather than 1s to represent edges. The recommended weighting function is 
 

Create a connected_components function, which takes in your graph’s list of nodes, and returns a list of lists – each inner list contains all nodes with a common label.

Create a propagate_label function that takes in a node, the node’s neighbors, and the adjacency matrix. It should update that node’s label based on the weights of its neighbor’s labels

Create a whispers function, which calls the propagate_label function some specified number of times on your graph. A node should be randomly selected each time the propagate_labels function is called.

While the whispers function is running – and updated the graph’s labels – consider using your connected_components function and record / plot how the number of connected components changes across iterations. At first each node should have its own unique label, and thus there should be as many connected components as nodes. But soon the number of connected components should decrease and converge to a stable number.

Gather a directory of pictures of, say, 10 distinct people. Each person should have 3-10 pictures. Create a graph from these pictures, run the whispers algorithm, and see if there is ultimately 10 connected components – one for each of the 10 distinct people.

Write code that will automatically organize the photos according to these connected component groupings