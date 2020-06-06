import csv
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Open datased and store both the attributes and species data into separate arrays
with open('leaf.csv', 'r') as leafs:
    leafs_reader = csv.reader(leafs, delimiter=',')
    attributes_array = []
    target_species = []
    for row in leafs_reader:
        target_species.append(row[0]) #Number of specie
        attributes_array.append(row[2:]) #Attibutes of the specimen

target_species = np.array(target_species, dtype=np.float32) #Convert from array to numpy array
attributes_array = np.array(attributes_array, dtype=np.float32) #Convert from array to numpy array

#Run the algoritm with the best k fit previously defined in the tests (K=20), check out the documentation to see also
#the explanation of the other algoritm parameters
knn = KNeighborsClassifier(algorithm='auto', n_neighbors=20, metric='braycurtis')

#Now it trains itself with the .csv data
knn.fit(attributes_array, target_species)

print("Model score: " + str(knn.score(attributes_array, target_species))) #overall model score with the training data

#Define all the possible classes that the result of the algoritm may belong to (Name of all the possible species)
classes = {1: 'Quercus suber', 2: 'Salix atrocinera', 3: 'Populus nigra', 4: 'Alnus sp.',5: 'Quercus robur',
           6: 'Crataegus monogyna', 7: 'Ilex aquifolium', 8: 'Nerium oleander', 9: 'Betula pubescens', 10: 'Tilia tomentosa',
           11: 'Acer palmatum', 12: 'Celtis sp.', 13: 'Corylus avellana', 14: 'Castanea sativa', 15: 'Populus alba',
           16: 'Acer negundo', 17: 'Taxus bacatta', 18: 'Papaver sp.', 19: 'Polypolium vulgare', 20: 'Pinus sp.',
           21: 'Fraxinius sp.', 22: 'Primula vulgaris', 23: 'Erodium sp.', 24: 'Bougainvillea sp.', 25: 'Arisarum vulgare',
           26: 'Euonymus japonicus', 27: 'Ilex perado ssp. azorica', 28: 'Magnolia soulangeana', 29: 'Buxus sempervirens',
           30: 'Urtica dioica', 31: 'Podocarpus sp.', 32: 'Acca sellowiana', 33: 'Hydrangea sp.', 34: 'Pseudosasa japonica',
           35: 'Magnolia grandiflora', 36: 'Geranium sp.'}


#Test input

x_new = [[0.98, 6.02, 0.83, 0.96, 1, 0.29, 0.024, 0.011, 0.03, 0.09, 0.009, 0.0027, 0.0002, 0.85]] #Very similar to the #72 row of the dataset. Expected specie: Nerium oleander

#Other tests
#x_new = [[0.88, 2.3, 0.6, 0.98, 0.99, 0.66, 0.019, 0.048, 0.023, 0.083, 0.007, 0.0013, 0.00027, 0.9]] #Slightly different from the row #20 of the dataset. Expected specie: Salix atrocinera
#x_new = [[0.46, 1.24, 0.39, 0.86, 1, 0.49, 0.031, 0.15, 0.1, 0.21, 0.04, 0.01, 0.0008, 1.76]] #Slightly different from the row #90 of the dataset. Expected specie: Betula pubescens
#x_new = [[0.87, 2.3, 0.52, 0.91, 0.99, 0.56, 0.006, 0.003, 0.03, 0.13, 0.015, 0.006, 0.00021, 1.23]] #Almost all attributes similar the row #90 of the dataset, last attribute moderately different. Expected specie: Acca sellowiana


#Run the classification
y_predict = knn.predict(x_new)
result = classes[y_predict[0]]

print("FINAL PREDICTION: " + result)

img=mpimg.imread('images/' + result + "/1.JPG")
imgplot = plt.imshow(img)
plt.xlabel(result)
plt.show()





