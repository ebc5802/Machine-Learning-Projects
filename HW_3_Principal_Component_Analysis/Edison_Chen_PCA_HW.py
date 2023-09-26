import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

print("\nStart of python program ^_^")

######### Load the data ##########

infile = open('faces.csv','r')
img_data = infile.read().strip().split('\n')
img = [map(int,a.strip().split(',')) for a in img_data]
pixels = []
for p in img:
    pixels += p
faces = np.reshape(pixels,(400,4096))

######### Global Variable ##########

image_count = 0
total_num_faces = 400

######### Display first face #########

# Useful functions:
# > numpy.reshape(a, newshape, order='C')
#   Gives a new shape to an array without changing its data.
# > matplotlib.pyplot.figure()
# 	Creates a new figure.
# > matplotlib.pyplot.title()
#	Set a title of the current axes.
# > matplotlib.pyplot.imshow()
#	Display an image on the axes.
#	Note: You need a matplotlib.pyplot.show() at the end to display all the figures.

first_face = np.reshape(faces[0],(64,64),order='F')
# Show first face
image_count+=1
plt.figure(image_count)
plt.title('first face')
plt.imshow(first_face,cmap=plt.cm.gray)
plt.show()

########## Display a random face ###########

# Useful functions:
# > numpy.random.choice(a, size=None, replace=True, p=None)
#   Generates a random sample from a given 1-D array
# > ndarray.reshape(shape, order='C')
#   Tuple of array dimensions.
#   Note: There are two ways to order the elements in an array: 
#         column-major order and row-major order. In np.reshape(), 
#         you can switch the order by order='C' for row-major (default), 
#         or by order='F' for column-major. 
# > numpy.reshape(a, newshape, order='C') 
#   Note: Equivalent function to numpy.reshape(a, newshape, order='C').


random_int = np.random.randint(0, total_num_faces-1)
random_face = np.reshape(faces[random_int],(64,64),order='F')
# Show random face
image_count+=1
plt.figure(image_count)
plt.title('random face #' + str(random_int))
plt.imshow(random_face,cmap=plt.cm.gray)
plt.show()




########## compute and display the mean face ###########

# Useful functions:
# > numpy.mean(a, axis='None', ...)
#   Compute the arithmetic mean along the specified axis.
#   Returns the average of the array elements. The average is taken over 
#   the flattened array by default, otherwise over the specified axis. 
#   Note: As a sanity check you might want to print the shape of the mean face 
#   and make sure that is equal to (4096,).

mean_face_raw = np.mean(faces, axis=0)
print("Shape of mean face:", mean_face_raw.shape)
mean_face = np.reshape(mean_face_raw,(64, 64),order='F')
# Show mean face
image_count+=1
plt.figure(image_count)
plt.title('mean face')
plt.imshow(mean_face,cmap=plt.cm.gray)
plt.show()




######### substract the mean from the face images and get the centralized data matrix A ###########

# Useful functions:
# > numpy.repeat(a, repeats, axis=None)
#   Repeat elements of an array.
#   Note: Faces is of shape (400, 4096) and mean face is of shape (4096, ).
#   As a side note you might wish to compute the centralized data matrix A without
#   using numpy.repeat(a, repeats, axis=None), but instead employing the operation 
#   of broadcasting that numpy offers.


#### YOUR CODE HERE ####
# print("faces:", faces)
# print("mean face raw:", mean_face_raw)

# Normalized faces
A = faces - mean_face_raw

# print("a:", a)


######### calculate the covariance matrix as is described in question 3 #####################

# Useful functions:
# > numpy.dot(a, b, out=None)
# Dot product of two arrays a and b. If a and b are 2-D array like in our case:
# > numpy.matmul() or @ is preferred over numpy.dot() to multiply a and b   
# > ndarray.T
#   Returns a view of the array with axes transposed.
#   Note: As a sanity check you might want to print the shape of the average
#   covariance matrix and make sure that it is equal to (4096, 4096). 

N = A.shape[0]
cov_matrix = (1/N) * np.dot(A.T, A)
# print("cov matrix, shape:", cov_matrix, cov_matrix.shape)



######### calculate the eigenvalues and eigenvectors of matrix L from the tutorial #####################

# > numpy.linalg.eig(a)[source]
#   Compute the eigenvalues and right eigenvectors of a square array.
#   The eigenvalues, each repeated according to its multiplicity. 
#   The eigenvalues are not necessarily ordered. 
#   As a sanity check print the shape of the eigenvectors and eigenvalues array
#   and make sure that are equal to (400,) and (400, 400) respectively. On the 
#   hand L is a (400, 400)-matrix too.


#### YOUR CODE HERE ####
print("A shape:", A.shape)
print("cov_matrix shape:", cov_matrix.shape)
# eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
# print("eigenvalues:", eigenvalues)
# print("eigenvectors:", eigenvectors)
# print("eigenvalues shape:", eigenvalues.shape)
# print("eigenvectors shape:", eigenvectors.shape)

L_matrix = np.dot(A, A.T)
eig_values, eig_vectors_L = np.linalg.eig(L_matrix)
print("eigenvalues:\n", eig_values)
print("eigenvalues shape:\n", eig_values.shape)
print("eigenvectors_L shape:\n", eig_vectors_L.shape)


######### compute the eigenvectors of V from the tutorial through the relation that connects them with those of L #####################

# > numpy.linalg.norm(x, ord=None, axis=None, keepdims=False) 
#   This function is able to return one of eight different matrix norms, 
#   or one of an infinite number of vector norms (described below), 
#   depending on the value of the ord parameter.
#   Note: in the given function, U should be a vector, not a array. 
#         You can write your own normalize function for normalizing 
#         the columns of an array.

# U can be an array
def normalize(U):
	return U / LA.norm(U, axis=0) # Normalizing the columns of an array

eig_vectors = np.dot(A.T, eig_vectors_L)
eig_vectors = normalize(eig_vectors)
print("eigenvectors:\n", eig_vectors)

# print("eigenvalues:", eig_values)
# print("eigenvectors:", eig_vectors)
# print("eigenvalues shape:", eig_values.shape)
print("normalized eigenvectors shape:", eig_vectors.shape)

# Sort eigenvalues and eigenvectors by descending eigenvalues
sorted_indices = np.argsort(eig_values)[::-1]
eig_values = eig_values[sorted_indices]
eig_vectors = eig_vectors[:, sorted_indices]



########## Display the first 16 principal components ##################
first_16_pc = eig_vectors[:, :16]

# Show first 16 principal components
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    pc = first_16_pc[:, i]
    pc_image = np.reshape(pc, (64, 64), order='F')
    ax.imshow(pc_image, cmap=plt.cm.gray)
    ax.set_title(f"Principal Component {i+1}")
plt.tight_layout()
plt.show()


########## Reconstruct the first face using the first two PCs #########
z1 = eig_vectors[:, 0]
z2 = eig_vectors[:, 1]
first_face_2_pc_raw = np.dot(A[0],z1) * z1 + np.dot(A[0],z2) * z2 + mean_face_raw
first_face_2_pc = np.reshape(first_face_2_pc_raw,(64,64),order='F')
image_count+=1
plt.figure(image_count)
plt.title('first face 2 PCs')
plt.imshow(first_face_2_pc,cmap=plt.cm.gray)
plt.show()




########## Reconstruct random face using the first 5, 10, 25, 50, 100, 200, 300, 399  PCs ###########
def reconstruct(random_face, pc_limit):
    random_face_raw = mean_face_raw
    for i in range(pc_limit):
        z_temp = eig_vectors[:, i]
        random_face_raw += np.dot(random_face,z_temp) * z_temp
    random_face_reconstructed = np.reshape(random_face_raw,(64,64),order='F')
    return random_face_reconstructed

random_index = np.random.randint(0, total_num_faces)
number_of_pc = [5, 10, 25, 50, 100, 200, 300, 399]

# Show face reconstructions
fig, axes = plt.subplots(4, 2, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    face_reconstructed_raw = reconstruct(A[random_index], number_of_pc[i])
    face_reconstructed = np.reshape(face_reconstructed_raw, (64, 64), order='F')
    ax.imshow(face_reconstructed, cmap=plt.cm.gray)
    ax.set_title(f"Face {random_index} w/ {number_of_pc[i]} PCs")
plt.tight_layout()
plt.show()



######### Plot proportion of variance of all the PCs ###############
plt.title("Principal Component Proportions of Variance")
numbered_list = list(range(1,401))
eig_values_adjusted = eig_values / sum(eig_values)
plt.plot(numbered_list, eig_values_adjusted, 'ro')
plt.show()




# More useful functions:
# > matplotlib.pyplot.plot(*args, **kwargs)
#   Plot lines and/or markers to the Axes. 
# > matplotlib.pyplot.show(*args, **kw)
#   Display a figure. 





#   When running in ipython with its pylab mode, 
#   display all figures and return to the ipython prompt.
