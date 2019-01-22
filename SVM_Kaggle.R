# Carga de datos
train = read.csv("./datos/train.csv", na.strings = c(" ", "NA", "?"))
test = read.csv("./datos/test.csv", na.strings = c(" ", "NA", "?"))

datos = train

# Imputacion de datos perdidos
cat("Datos completos: ", ncc(datos), " e incompletos: ", nic(datos), "\n")
imputados <- mice(datos)
datos <- complete(imputados)

# TO DO: Normalize data (Test?)
#datosPre <- caret::preProcess(datos[,1:50],method=c("center","scale"))
#datosTransformados <- predict(datosPre,datos[,1:50])
#datos[ , -51] <- datosTransformados

#testPre <- caret::preProcess(test,method=c("center","scale"))
#test <- predict(testPre, test)


# Filtrado de variables muy correladas
library(corrplot)
library(caret)
corrMatrix = cor(na.omit(datos))
corrplot::corrplot(corrMatrix, 
                   order = "FPC", 
                   type = "upper", 
                   tl.col = "black", 
                   tl.str = 45) # FPC ordena de izquierda a derecha las mas correladas

altaCorrelacion = caret::findCorrelation(corrMatrix, cutoff = 0.8)
datos = datos[ , -altaCorrelacion]

# TO DO: tune best parameters (C and Gamma) -> tune.svm()

# Generacion del modelo con SVM
library(e1071)
model = e1071::svm(C ~ ., data = datos, type = "C-classification")

# Generacion de la prediccion
pred = predict(model, test)

# Guarda el archivo para Kaggle
ids = 1:nrow(test)
res = data.frame(ids, pred)
colnames(res) = c("Id", "Prediction")

write.csv(res, file = "./datos/resultado.csv", row.names = FALSE, quote = FALSE)
