# Exercise 1: Text Similarity using Hamming Distance

Concept: Hamming distance measures the number of differing bits or characters between two strings of equal length.

Practical Application: Comparing DNA sequences, error detection in communication, and simple text similarity.

Task:
1.  Write a Python function `hamming_distance(str1, str2)` that calculates the Hamming distance between two strings.
2.  Use the function to compare the following pairs of strings and print their Hamming distances:
    a. "karolin" and "kathrin"
    b. "toned" and "roses"
    c. "1011101" and "1001001"
3.  Discuss how the Hamming distance reflects the similarity between the strings in each pair.

~~~
def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must be of equal length")
    distance = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            distance += 1
    return distance

print(hamming_distance("karolin", "kathrin"))
print(hamming_distance("toned", "roses"))
print(hamming_distance("1011101", "1001001"))
~~~

# Exercise 2: Image Feature Comparison using Euclidean Distance

Concept: Euclidean distance is the straight-line distance between two points in Euclidean space.

Practical Application: Image recognition, clustering, and nearest neighbor searches based on feature vectors.

Task:
1.  Imagine you have two images, and you've extracted feature vectors from them (e.g., using color histograms or deep learning embeddings).
2.  Let the feature vectors be:
    a. Image 1: `[1.2, 3.5, 2.1, 0.8]`
    b. Image 2: `[1.0, 3.0, 2.5, 1.0]`
3.  Write a Python function `euclidean_distance(vec1, vec2)` that calculates the Euclidean distance between two vectors.
4.  Calculate and print the Euclidean distance between the two image feature vectors.
5.  Discuss how a smaller Euclidean distance implies higher similarity between the images.

~~~
import math

def euclidean_distance(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of equal length")
    squared_diffs = [(vec1[i] - vec2[i]) for i in range(len(vec1))]
    return math.sqrt(sum(squared_diffs))

vec1 = [1.2, 3.5, 2.1, 0.8]
vec2 = [1.0, 3.0, 2.5, 1.0]

print(euclidean_distance(vec1, vec2))
~~~

# Exercise 3: Location-Based Recommendations using Manhattan Distance

Concept: Manhattan distance (or L1 distance) is the sum of the absolute differences between the coordinates of two points.

Practical Application: Route planning, urban distance calculations, and recommendation systems based on location.

Task:
1.  You have a dataset of restaurants and their coordinates (latitude, longitude).
2.  Your current location is (34.0522, -118.2437) (Los Angeles).
3.  The coordinates of three restaurants are:
    a. Restaurant A: (34.0300, -118.2600)
    b. Restaurant B: (34.0700, -118.2200)
    c. Restaurant C: (34.1000, -118.3000)
4.  Write a Python function `manhattan_distance(coord1, coord2)` that calculates the Manhattan distance between two coordinate pairs.
5.  Calculate the Manhattan distance from your location to each restaurant.
6.  Recommend the restaurant with the smallest Manhattan distance as the closest.

~~~
def manhattan_distance(coord1, coord2):
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

my_location = (34.0522, -118.2437)
restaurant_a = (34.0300, -118.2600)
restaurant_b = (34.0700, -118.2200)
restaurant_c = (34.1000, -118.3000)

distance_a = manhattan_distance(my_location, restaurant_a)
distance_b = manhattan_distance(my_location, restaurant_b)
distance_c = manhattan_distance(my_location, restaurant_c)

print(f"Distance to A: {distance_a}")
print(f"Distance to B: {distance_b}")
print(f"Distance to C: {distance_c}")

closest_restaurant = min([(distance_a, "A"), (distance_b, "B"), (distance_c, "C")])[1]
print(f"The closest restaurant is: {closest_restaurant}")
~~~

# Exercise 4: General Distance Calculation with Minkowski Distance

Concept: Minkowski distance is a generalization of Euclidean and Manhattan distances, with a parameter 'p' controlling the distance metric.

Practical Application: Flexible distance measure for various data types and applications, adaptable to different norms.

Task:
1.  Write a Python function `minkowski_distance(vec1, vec2, p)` that calculates the Minkowski distance between two vectors.
2.  Use the function to calculate the distance between vectors `[1, 2, 3]` and `[4, 5, 6]` for:
    a. p = 1 (Manhattan distance)
    b. p = 2 (Euclidean distance)
    c. p = 3 (a general Minkowski distance)
3.  Observe how the distance changes with different values of 'p'.

~~~
import numpy as np
def minkowski_distance(vec1, vec2, p):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of equal length")
    abs_diffs = [abs(vec1[i] - vec2[i]) for i in range(len(vec1))]
    return np.power(sum(np.power(abs_diffs, p)), 1/p)

vec1 = [1, 2, 3]
vec2 = [4, 5, 6]

print(f"Minkowski distance (p=1): {minkowski_distance(vec1, vec2, 1)}")
print(f"Minkowski distance (p=2): {minkowski_distance(vec1, vec2, 2)}")
print(f"Minkowski distance (p=3): {minkowski_distance(vec1, vec2, 3)}")
~~~


![iWe4J](https://github.com/user-attachments/assets/15379184-a920-47b7-aa21-ef6ef6a4bc7c)










# Exercise 5: Hamming Distance for Categorical Data Comparison

Objective: Implement a function to calculate the Hamming distance between two strings (representing categorical feature vectors) and apply it to compare customer profiles.

~~~
def hamming_distance(str1, str2):
  # Calculates the Hamming distance between two strings.
  if len(str1) != len(str2):
    raise ValueError("Strings must be of equal length.")

  distance = 0
  for char1, char2 in zip(str1, str2):
    if char1 != char2:
      distance += 1
  return distance

customer1 = "ABACD"
customer2 = "AABCE"
customer3 = "BBACD"

print(f"Hamming distance between customer1 and customer2: {hamming_distance(customer1, customer2)}")
print(f"Hamming distance between customer1 and customer3: {hamming_distance(customer1, customer3)}")
~~~



# Exercise 6: Euclidean and Manhattan Distance for Numerical Data

Objective: Implement functions to calculate Euclidean and Manhattan distances and compare their results on a small dataset of points.

~~~
import numpy as np

def euclidean_distance(point1, point2):
  # Calculates the Euclidean distance between two points.
  point1 = np.array(point1)
  point2 = np.array(point2)
  return np.linalg.norm(point1 - point2)

def manhattan_distance(point1, point2):
  # Calculates the Manhattan distance between two points.
  point1 = np.array(point1)
  point2 = np.array(point2)
  return np.sum(np.abs(point1 - point2))

point_a = (1, 2)
point_b = (4, 6)
point_c = (1, 5)

print(f"Euclidean distance between A and B: {euclidean_distance(point_a, point_b)}")
print(f"Manhattan distance between A and B: {manhattan_distance(point_a, point_b)}")
print(f"Euclidean distance between A and C: {euclidean_distance(point_a, point_c)}")
print(f"Manhattan distance between A and C: {manhattan_distance(point_a, point_c)}")
~~~

# Exercise 7: Minkowski Distance with Varying 'p' Values

Objective: Implement the Minkowski distance function and observe how the distance changes with different 'p' values (p=1 for Manhattan, p=2 for Euclidean).

~~~
import numpy as np

def minkowski_distance(point1, point2, p):
  # Calculates the Minkowski distance between two points.
  point1 = np.array(point1)
  point2 = np.array(point2)
  return np.power(np.sum(np.power(np.abs(point1 - point2), p)), 1/p)

point_d = (0, 0)
point_e = (3, 4)

print(f"Minkowski distance (p=1, Manhattan): {minkowski_distance(point_d, point_e, 1)}")
print(f"Minkowski distance (p=2, Euclidean): {minkowski_distance(point_d, point_e, 2)}")
print(f"Minkowski distance (p=3): {minkowski_distance(point_d, point_e, 3)}")
print(f"Minkowski distance (p=4): {minkowski_distance(point_d, point_e, 4)}")
~~~

# Exercise 8: Cosine Distance for Text Similarity

Objective: Calculate the cosine distance between two text documents represented as TF-IDF vectors.

~~~ 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate cosine distances
cosine_dist_1_2 = cosine_distances(tfidf_matrix[0], tfidf_matrix[1])[0][0]
cosine_dist_1_4 = cosine_distances(tfidf_matrix[0], tfidf_matrix[3])[0][0]

print(f"Cosine distance between doc1 and doc2: {cosine_dist_1_2}")
print(f"Cosine distance between doc1 and doc4: {cosine_dist_1_4}")

~~~

# Exercise 9: Combining Distances for Hybrid Similarity

Objective: Create a hybrid distance metric that combines Euclidean and Cosine distances for data that has both numerical and textual features.

~~~
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

# Sample data with numerical and textual features
data = [
    {"numeric": [1, 2], "text": "apple banana"},
    {"numeric": [3, 4], "text": "banana cherry"},
    {"numeric": [1, 5], "text": "apple cherry"},
]

def hybrid_distance(data1, data2, alpha=0.5):
  # Calculates a hybrid distance using Euclidean and Cosine distances.
  numeric_dist = euclidean_distance(data1["numeric"], data2["numeric"])

  vectorizer = TfidfVectorizer()
  tfidf_matrix = vectorizer.fit_transform([data1["text"], data2["text"]])
  cosine_dist = cosine_distances(tfidf_matrix[0], tfidf_matrix[1])[0][0]

  return alpha * numeric_dist + (1 - alpha) * cosine_dist

# Calculate hybrid distances
dist_1_2 = hybrid_distance(data[0], data[1])
dist_1_3 = hybrid_distance(data[0], data[2])

print(f"Hybrid distance between data1 and data2: {dist_1_2}")
print(f"Hybrid distance between data1 and data3: {dist_1_3}")

~~~
