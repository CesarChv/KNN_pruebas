import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_svmlight_file

archivo = 'C:/Users/zo0i_/Desktop/Programas_python/Knn_python_programa_2/Iris_DB_svm.txt'
x_train, y_train = load_svmlight_file(archivo)

print(x_train)     #Grupo de entrenamiento
print(y_train) 

print(type(x_train))     #Grupo de entrenamiento
print(type(y_train)) 
#train_test_split
#regreso grupos entrenamiento y pruebas, recibe datos y clasificacion
x_entrenar, x_test, y_entrenar, y_test = train_test_split(x_train, y_train)
#Toma como referencia 75% los datos de entrenamiento
print(x_entrenar.shape)     #Grupo de entrenamiento
print(y_entrenar.shape)     #Grupo de entrenamiento de clase

#clasidicador
knn = KNeighborsClassifier( n_neighbors=10 )

knn.fit(x_entrenar, y_entrenar)        #Entrenamiento

aprend = knn.score(x_test, y_test)      #Prubas(aprendizaje)
print('{:.5f}'.format(aprend) )

clasif = knn.predict([[1.2, 3.4, 0.6, 1.1]])    #nuevo patron
print(clasif)
clasif1 = knn.predict([[0.2, 3.3, 1.5, 0.1]]) 
print(clasif)
clasif = knn.predict([[2, 4.1, 3.6, 5.9]]) 
print(clasif)
clasif = knn.predict([[2.2, 2.4, 2.6, 0.9]]) 
print(clasif)
clasif = knn.predict([[1.2, 4.1, 3.6, 5.9]]) 
#print(clasif,'=', db_iris['target_names'][clasif])


if clasif == 0:
	print(clasif, "= Setosa")
elif clasif == 1:
	print(clasif, "= Versicolor")
elif clasif == 2:
	print(clasif, "= Virginica")
else:
	print(clasif, "= Tulipan")