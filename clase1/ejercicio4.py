import numpy as np

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
truth = np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 1])  # Verdad
print("Verdadero ", truth)
prediction = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0])  # Predicción
print("Predicción", prediction, "\n")
# valor esperado para TP ([1 1 0 1 0 0 0 0 0 0])
# valor esperado para TN ([0 0 0 0 0 0 0 0 1 0])
# valor esperado para TP ([0 0 0 0 1 1 0 0 0 1])
# valor esperado para TN ([0 0 1 0 0 0 1 1 0 0])

print("Valor esperado para True Positive ", "[1 1 0 1 0 0 0 0 0 0]")
truePositive = (truth == 1) & (prediction == 1)
# truePositive =  truth & prediction
print("Valor obtenido para True Positive ", truePositive.astype(int))

print("Valor esperado para True Negative ", "[0 0 0 0 0 0 0 0 1 0]")
trueNegative = (truth == 0) & (prediction == 0)
# trueNegative =  truth & prediction
print("Valor obtenido para True Negative ", trueNegative.astype(int))

print("Valor esperado para False Negative", "[0 0 0 0 1 1 0 0 0 1]")
falseNegative = (truth == 1) & (prediction == 0)
print("Valor obtenido para False Negative", falseNegative.astype(int))

print("Valor esperado para False Positive", "[0 0 1 0 0 0 1 1 0 0]")
falsePositive = (truth == 0) & (prediction == 1)
print("Valor obtenido para False Positive", falsePositive.astype(int))

precision = np.sum(truePositive) / (np.sum(truePositive) + np.sum(falsePositive))
print("Precision = TP / (TP + FP) = ", precision)

recall = np.sum(truePositive) / (np.sum(truePositive) + np.sum(falseNegative))
print("Recall = TP / (TP + FN) = ", recall)

accuracy = (np.sum(truePositive) + np.sum(trueNegative)) / (
np.sum(truePositive) + np.sum(trueNegative) + np.sum(falsePositive) + np.sum(falseNegative))
print("Accuracy = (TP + TN) / (TP + TN + FP + FN) = ", accuracy)
