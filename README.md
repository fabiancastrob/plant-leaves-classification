# Python Plant Leaves Classifier - Problem Description

This python program classifies plant leaves according to several attributes. The goal is to locate the plant leaf in a predefined species, according to the input parameters.

It uses the K-Nearest Neighbors algorithm from python's scikit-learn library. At the end of the execution, the user will get the classification of the leaf as well as an image of a similar leaf, according to the species.

In the actual context, we may find several applications for classification problems using Machine Learning techniques. From medicine applications using image recognition to biological classifications like this example. The wide range of useful scenarios where the use of cutting edge technology is more than justified makes this area of Computer Science one of the promises for the future.

## Requisites
You will need to install Python along with the following libraries:
* Python
* Pillow
* Matplotlib
* Numpy
* Scikit-learn
* Scipy

## Steps
* Clone the repository
* Open a terminal in the repository's location in your computer
* Run the program with the following command:
```python main.py```


## Why K-Nearest Neighbors?
The nature of the problem solved in this context is a classification problem. Hence, with a proper training data, good results can be obtained though the use of the K-Nearest Neighbors algorithm.

The negative aspects in this case are: there will be no learning process and the computing processing needed can be a little high. However, for this real life example, we have all the resources needed.


## The data
The data set to be used within this solution has a total of 36 different species of plant leaves classified according to several parameters. In total, there are 340 records of different specimens.

In the following figure, we will take a look of how the data looks like, in the first 5 records.

![alt text](https://github.com/fabiancastrob/plant-leaves-classification/blob/master/images/data.png?raw=true)

In total, there are 16 parameters for each record, which describe both shape and texture for every leaf:
* A : Class (Species)
* B: Specimen number
* C: Eccentricity
* D: Aspect ratio
* E: Elongation
* F: Solidity
* G: Stochastic Convexity
* H: Isoperimetric Factor
* I: Maximal Indentation Depth
* J: Lobedness
* K: Average Intensity
* L: Average Contrast
* M: Smoothness
* N: Third moment
* O: Uniformity
* P: Entropy

Also, the possible species for classification are:
* Quercus suber
* Salix atrocinera
* Populus nigra
* Alnus sp.
* Quercus robur
* Crataegus monogyna
* Ilex aquifolium
* Nerium oleander
* Betula pubescens
* Tilia tomentosa
* Acer palmatum
* Celtis sp.
* Corylus avellana
* Castanea sativa
* Populus alba
* Acer negundo
* Taxus bacatta
* Papaver sp.
* Polypolium vulgare
* Pinus sp.
* Fraxinius sp.
* Primula vulgaris
* Erodium sp.
* Bougainvillea sp.
* Arisarum vulgare
* Euonymus japonicus
* Ilex perado ssp. azorica
* Magnolia soulangeana
* Buxus sempervirens
* Urtica dioica
* Podocarpus sp.
* Acca sellowiana
* Hydrangea sp.
* Pseudosasa japonica
* Magnolia grandiflora
* Geranium sp.

The main idea is to input a complete vector of characteristics and to get in the output the correspondent classification, as well as a picture of one of the specimens of the species.


## Why K value = 20?
Several discussions talk about how to choose a proper k value for the K-Nearest Neighbors algorithm. For this case, an iteration was run to determine the biggest change in the chart. From the range from 1 to 40, here are the results obtained.

![alt text](https://github.com/fabiancastrob/plant-leaves-classification/blob/master/images/k_chart.png?raw=true)

From the first k values, we see a tendency to get a lower accuracy, however, low K values might tend to be disguising for the algorithm because the scope might not be the most adequate. The best "elbow effect" we can get is around k = 20. In this case, although the accuracy is poorer, we have a bigger spectrum of items to be compared with. Of course we are sacrificing more resources for a more accurate output.

## K-Nearest Neighbors parameters explanation
Making use of the scikit-learn's K-Nearest Neighbors classifier simplifies a lot of job, however, it needs to be very carefully parameterized.

### Neighbors choose algorithm

This is the way the algorithm searches for the neighbors of a specific record. With scikit-learn, we have the ball_tree, kd_tree, brute and auto. They are better or worse depending on the data set naturalness, however, the auto option tries to choose the best approach for the specific case. Hence, we specified
```algorithm = 'auto'```


### K value
Previously justified. This is the parameter that specifies the number of neighbors for the input to be compared with.

### Distance Metric
This is the way that the algorithm measures the distances with the neighbors. After trying to find the optimal, the BrayCurtis metric was choosen. Given the naturalness of the data, better model accuracies were obtained.

## References
Dudani, S. A. (1976). The distance-weighted k-nearest-neighbor rule. IEEE Transactions on Systems, Man, and Cybernetics, (4), 325-327.
Hu, L., Huang, M., Ke, S. et al. The distance function effect on k-nearest neighbor classification for medical datasets.
Enas, G. G., & Choi, S. C. (1986). Choice of the smoothing parameter and efficiency of k-nearest neighbor classification. In Statistical Methods of Discrimination and Classification (pp. 235-244). Pergamon.
