# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

print("---- Ejercicio 1 ----")
print("Computar las normas l0, l1, l2, l-infinito\n")
a = np.array ([[5,9,7],[1,2,-2],[3,5,-6]])

print("Matriz A = \n", a)

def norma0(a):
    a = (np.sum(a, axis=1))
    return a
    
def norma1(a):
    a = np.absolute(a)
    a = np.sum(a, axis=1)
    return a

def norma2(a):
    a = np.sum((a**2), axis=1)**(1/2)
    return a
    
def normainf(a):
    a = np.amax(np.absolute(a))
    return a
    


print("Norma 0", norma0(a))
print("Norma 1", norma1(a))
print("Norma 2", norma2(a))
print("Norma inf", normainf(a))

print("\n---- Ejercicio 2 ----")
print("Computar la norma l2 de cada vector, ordenar de mayor a menor\n")


print("Matriz B = \n", a)
print("Norma 2\n", norma2(a))
b = -np.sort(-norma2(a))
print("Norma 2 ordena\n", b)


print("\n---- Ejercicio 3 ----")
print("Construir un índice para identificadores de usuarios, id2idx e idx2id\n")

'''
El objetivo es construir un índice para identificadores de usuarios,
es decir id2idx e idx2id. Para ello crear una clase, donde el
índice se genere en el constructor. Armar métodos get_users_id 
y get_users_idx.

Identificadores de usuarios : users_id = [15, 12, 14, 10, 1, 2, 1]
Índice de usuarios : users_id = [0, 1, 2, 3, 4, 5, 4]

id2idx =  [-1     4     5    -1    -1    -1     -1    -1    -1    -1     3     -1      1    -1     2     0]
          [ 0     1     2     3     4     5      6     7     8     9    10     11     12    13    14    15]

id2idx[15] -> 0 ; id2idx[12] -> 1 ; id2idx[3] -> -1
idx2id[0] -> 15 ; idx2id[4] -> 1

'''

class Usuario:
    
    instance = None
    
    def __new__(self, id2id):
        if Usuario.instance is None:
            print("__new__ usuario creado")
            Usuario.instance = super(Usuario, self).__new__(self)
            return Usuario.instance
        else:
            return Usuario.instance
        
    def __init__(self, id2id):
        print("__init__")
        self.id2id = id2id
        self.idx = np.full(shape = np.max(id2id) + 1, fill_value = -1)   
        print("valores", id2id)
        #u, indices = np.unique(id2id, return_index=True)
        mylist = list(dict.fromkeys(id2id))
        print("valores", mylist)
        print(type(mylist))
        j = 0
        for i in mylist:
            self.idx[i] = j
            j = j + 1
        print("Idx \n",self.idx)
        
    def get_user_id(self,id2):
        return(self.id2id[id2])
    def get_user_idx(self,id2):
        return (self.idx[id2])

        
id2id = np.array([15, 12, 14, 10, 1, 2, 1])
user_1 = Usuario(id2id)

print("Get user id  2 =", user_1.get_user_id(2))
print("Get user idx 4 =", user_1.get_user_idx(5))

print(hex(id(user_1)))

print("\n---- Ejercicio 4 ----")
print("Precision, Recall, Accuracy\n")
print("True Positive (TP): la verdad es 1 y la predicción es 1.")
print("True Negative (TN): la verdad es 0 y la predicción es 0.")
print("False Negative (FN): la verdad es 1 y la predicción es 0.")
print("False Positive (FP): la verdad es 0 y la predicción es 1.\n")
'''--
En los problemas de clasificación, se cuenta con dos arreglos, la verdad (ground truth) y la predicción (prediction). 
Cada elemento de los arreglos puede tomar dos valores: 
True (representado por 1) y False (representado por 0). 
Por lo tanto, se pueden definir cuatro variables:   
     
True Positive (TP): la verdad es 1 y la predicción es 1.
True Negative (TN): la verdad es 0 y la predicción es 0.
False Negative (FN): la verdad es 1 y la predicción es 0.
False Positive (FP): la verdad es 0 y la predicción es 1.
A partir de esas cuatro variables, se definen las siguientes métricas:

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Para los siguientes arreglos, representando la verdad y la predicción, calcular las métricas anteriores con 
operaciones vectorizadas en NumPy.

truth = [1,1,0,1,1,1,0,0,0,1]
prediction = [1,1,1,1,0,0,1,1,0,0] '''

print("Vectores : \n ")
truth        = np.array ([1,1,0,1,1,1,0,0,0,1])            # Verdad
print("Verdadero ", truth)
prediction   = np.array ([1,1,1,1,0,0,1,1,0,0])       # Prediccion
print("Prediccion", prediction, "\n")
#valor esperado para TP ([1 1 0 1 0 0 0 0 0 0])
#valor esperado para TN ([0 0 0 0 0 0 0 0 1 0])
#valor esperado para TP ([0 0 0 0 1 1 0 0 0 1])
#valor esperado para TN ([0 0 1 0 0 0 1 1 0 0])

print("Valor esperado para True Positive ", "[1 1 0 1 0 0 0 0 0 0]")
truePositive =  (truth == 1) & (prediction == 1)
#truePositive =  truth & prediction
print("Valor obtenido para True Positive ", truePositive.astype(int))

print("Valor esperado para True Negative ", "[0 0 0 0 0 0 0 0 1 0]")
trueNegative =  (truth == 0) & (prediction == 0)
#trueNegative =  truth & prediction
print("Valor obtenido para True Negative ", trueNegative.astype(int))

print("Valor esperado para False Negative", "[0 0 0 0 1 1 0 0 0 1]")
falseNegative =  (truth == 1) & (prediction == 0)
print("Valor obtenido para False Negative", falseNegative.astype(int))

print("Valor esperado para False Positive", "[0 0 1 0 0 0 1 1 0 0]")
falsePositive =  (truth == 0) & (prediction == 1)
print("Valor obtenido para False Positive", falsePositive.astype(int))

precision = np.sum(truePositive) / (np.sum(truePositive) + np.sum(falsePositive))
print("Precision = TP / (TP + FP) = ", precision)

recall = np.sum(truePositive) / (np.sum(truePositive) + np.sum(falseNegative))
print("Recall = TP / (TP + FN) = ", recall)

accuracy = (np.sum(truePositive) + np.sum(trueNegative)) / (np.sum(truePositive) + np.sum(trueNegative) + np.sum(falsePositive) + np.sum(falseNegative))
print("Accuracy = (TP + TN) / (TP + TN + FP + FN) = ", accuracy)

print("\n---- Ejercicio 5 ----")
print("Average Query Precision\n")

'''En information retrieval o search engines, en general contamos con queries “q” y para cada “q” una lista de documentos 
que son verdaderamente relevantes. Para evaluar un search engine, es común utilizar la métrica average query precision. 
Tomando de referencia el siguiente ejemplo, calcular la métrica con NumPy utilizando operaciones vectorizadas.

q_id =             [1, 1, 1, 1,  2, 2, 2,  3, 3, 3, 3, 3,  4, 4, 4, 4]
predicted_rank =   [0, 1, 2, 3,  0, 1, 2,  0, 1, 2, 3, 4,  0, 1, 2, 3]
truth_relevance =  [T, F, T, F,  T, T, T,  F, F, F, F, F,  T, F, F, T] 

Precision para q_id 1 = 2 / 4
Precision para q_id 2 = 3 / 3
Precision para q_id 3 = 0 / 5
Precision para q_id 4 = 2 / 4

average query precision = ((2/4) + (3/3) + (0/5) + (2/4)) / 4
'''

q_id            = np.array ([1, 1, 1, 1,      2, 2, 2,       3, 3, 3, 3, 3,       4, 4, 4, 4])   
predicted_rank  = np.array ([0, 1, 2, 3,      0, 1, 2,       0, 1, 2, 3, 4,       0, 1, 2, 3])   
#predicted_rank  = np.array([1, 0, 1, 0,      1, 1, 1,       0, 0, 0, 0, 0,       1, 0, 0, 1])  
truth_relevance = np.array (['T','F','T','F','T','T','T',  'F','F','F','F','F',  'T','F','F','T'])   

         
print("q_id             = ", q_id)
print("Predicted_rank   = ", predicted_rank)
print("Truth_relevance  = ", truth_relevance)

truth_relevance =  (truth_relevance == 'T') 

'''Creo que tengo que contar la cantidad de T por cada id
'''
auxTrue = np.zeros(np.amax(predicted_rank), dtype=int)
auxDatos = np.zeros(np.amax(predicted_rank), dtype=int)
for i in range(q_id.size):
    if truth_relevance[i]:
        auxTrue[q_id[i]-1] += 1 
    auxDatos[q_id[i]-1] += 1
print("auxTrue",auxTrue)
print("auxDatos",auxDatos)  
for i in range(np.amax(predicted_rank)):
    print("Precision para q_id ", i + 1, " = ", auxTrue[i], " / ", auxDatos[i], " = ", auxTrue[i]/auxDatos[i] )


print("\n---- Ejercicio 6 ----")
print("Distancia a Centroides\n")

'''
Dada una nube de puntos X y centroides C, obtener la distancia entre cada vector X y los centroides utilizando operaciones vectorizadas y broadcasting en NumPy. Utilizar como referencia los siguientes valores:

X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
C = [[1, 0, 0], [0, 1, 1]]   
'''
def dist(a,b):  
    distancia = np.sum((a-b)**2, axis=1)**(1/2) 
    return distancia

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Puntos     = \n", X)
C = np.array([[1, 0, 0], [0, 1, 1]])
print("Centroides = \n", C)

print("Distancia de los puntos en X a C[0] es = ", dist(X,C[0]))
print("Distancia de los puntos en X a C[1] es = ", dist(X,C[1]))

print("\n---- Ejercicio 7 ----")
print("Etiquetar Cluster\n")

'''
Obtener para cada fila en X, el índice de la fila en C con distancia euclídea más pequeña. Es decir, para cada fila en X, 
determinar a qué cluster pertenece en C. Hint: usar np.argmin.
'''
print("x = \n", X)
'''Obtener para cada punto en X el cluster mas cercano'''

print("Valores calculados ej 6 =  [ 3.6  8.3  13.4]")
print("                           [ 2.4  7.5  12.7]")
print("Resultado esperado         [ P[1] P[1] P[1]]")


def distancias_a_centroides(puntos, centroides):
    '''
    Parameters
     numpy.reshape(a, newshape, order='C')
    '''
    exppanded_C = centroides[:, None]
    distancias = np.sqrt(np.sum((exppanded_C - puntos) ** 2, axis=2)) 
    print("Distancias \n", distancias)
    return distancias

arg_min = np.argmin(distancias_a_centroides(X,C),axis=0)

print("Centroide de pertenencia \n", arg_min)

print("\n---- Ejercicio 8 ----")
print("Implementación Básica de K-means\n")

'''
K-means es uno de los algoritmos más básicos en Machine Learning no supervisado. 
Es un algoritmo de clusterización, que agrupa datos que comparten características similares. 
Recordemos que entendemos datos como n realizaciones del vector aleatorio X.

El algoritmo funciona de la siguiente manera:

 1_ El usuario selecciona la cantidad de clusters a crear n. OK
 2_ Se seleccionan n elementos aleatorios de X como posiciones iniciales del los centroides C.
 3_ Se calcula la distancia entre todos los puntos en X y todos los puntos en C.
 4_ Para cada punto en X se selecciona el centroide más cercano de C.
 5_ Se recalculan los centroides C a partir de usar las filas de X que pertenecen a cada centroide.
 6_ Se itera entre 3 y 5 una cantidad fija de veces o hasta que la posición de los centroides no cambie dada una tolerancia.
 
Se debe por lo tanto implementar la función k_means(X, n) de manera tal que, al finalizar, devuelva la posición de los 
centroides y a qué cluster pertenece cada fila de X.

Hint: para (2) utilizar funciones de np.random, para (3) y (4) usar los ejercicios anteriores, para (5) es válido 
utilizar un for. Iterar 10 veces entre (3) y (5).
'''

'''
datos = n realizaciones del vector aleatorio X

'''
MAX_ITERATIONS = 5

def k_means(X, n_clusters): # el usuario selecciona la cantidad de clusters a crear n 
    '''
    numpy.eye(N, M=None, k=0, dtype=<class 'float'>, order='C', *, like=None)
    Return a 2-D array with ones on the diagonal and zeros elsewhere.

    '''
    centroids = np.eye(n_clusters, X.shape[1])          # Se seleccionan n elementos aleatorios como posiciones iniciales de los centroides
    print("Centroides \n", centroids)
    for i in range(MAX_ITERATIONS):
        print("Iteration # {}".format(i))
        centroids, cluster_ids = k_means_loop(X, centroids)
        print(centroids)
    return centroids, cluster_ids

def k_means_loop(X, centroids):
    
    # encontrar el label de cada fila de X en funcion de los centroides
    expanded_centroids = centroids[:, None]
    distances = np.sqrt(np.sum((expanded_centroids - X) ** 2, axis=2))
    arg_min = np.argmin(distances, axis=0)
    #print("Distancias \n", distances)
    
    #rederterminar los centroides
    for i in range(centroids.shape[0]):
        centroids[i] = np.mean(X[arg_min == i, :], axis=0)
    #print("Centroides 2\n", centroids)    
    return centroids, arg_min
        
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]])

centroids, cluster_ids = k_means(X, n_clusters=2)

print("Nuevos centroides \n", centroids)
print("Pertenencia de los puntos  a los respectivos centroides\n", cluster_ids)






















 
 