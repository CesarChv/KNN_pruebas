
from distutils import archive_util
import sklearn
from sklearn.datasets import load_iris #cargar base de datos
from sklearn.model_selection import train_test_split #dividir y tomar porcentaje de 80 para entrenar y el resto para las pruebas """"
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_svmlight_file

db_iris = load_iris() #como se esta cargando la bd
print(type(db_iris))          #imprimir 

print(db_iris.keys()) # seguarda en la variable db_irirs
#data, target, target_names, decr, names

print(db_iris['data']) #imprimir los 150 datos con 4 elementos 

print(db_iris['target_names'])

print(db_iris['target']) #las clases de cada uno de mis elementos

print(db_iris['feature_names'])


# x son los datos, y son las clases, test para las pruebas

#------------------------------------------------------------------------------------------------------------------------------------------
x_entrenar, x_test, y_entrenar, y_test = train_test_split(db_iris['data'], db_iris['target'] )

print(x_entrenar.shape) #grupo de entrenamiento
print(y_entrenar.shape) #grupo de entramiento clases


#clasificar
knn = KNeighborsClassifier(n_neighbors = 10)

knn.fit(x_entrenar, y_entrenar)     #entrenamiento

aprend = knn.score(x_test, y_test)  #Pruebas aprendizaje
print ('{:.5f}'.format(aprend) )

#--------------------------------------------#--------------------------------------------#--------------------------------------------
#--------------------------------------------#--------------------------------------------#--------------------------------------------
clasif = knn.predict([[8.0, 4.0, 6.4, 2.0]])   #nuevo patron
print(clasif,' = ', db_iris['target_names'][clasif])
#-------------------------------------------------------------
clasif2 = knn.predict([[8.0, 2.0, 3.4, 5.0]])   #nuevo patron
print(clasif2,' = ', db_iris['target_names'][clasif2])
#-------------------------------------------------------------
clasif4 = knn.predict([[7.9, 8.0, 7.4, 0.2]])   #nuevo patron
print(clasif4,' = ', db_iris['target_names'][clasif4])
print("\n")
#--------------------------------------------#--------------------------------------------#--------------------------------------------
#--------------------------------------------#--------------------------------------------#--------------------------------------------
clasif3 = knn.predict([[7.0, 3.0, 4.4, 1.0]])   #nuevo patron
print(clasif3,' = ', db_iris['target_names'][clasif3])
#-------------------------------------------------------------
clasif4 = knn.predict([[7.0, 4.4, 3.4, 1.0]])   #nuevo patron
print(clasif4,' = ', db_iris['target_names'][clasif4])
print("\n")
#--------------------------------------------#--------------------------------------------#--------------------------------------------
#--------------------------------------------#--------------------------------------------#--------------------------------------------
clasif4 = knn.predict([[4.9, 3.0, 1.4, 0.2]])   #nuevo patron
print(clasif4,' = ', db_iris['target_names'][clasif4])
#-------------------------------------------------------------
clasif4 = knn.predict([[0.9, 3.0, 0.4, 0.2]])   #nuevo patron
print(clasif4,' = ', db_iris['target_names'][clasif4])

