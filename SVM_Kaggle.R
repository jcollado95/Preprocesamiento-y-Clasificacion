# Carga de datos
datos = read.csv("./datos/train.csv", na.strings = c(" ", "NA", "?"))
test = read.csv("./datos/test.csv", na.strings = c(" ", "NA", "?"))

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
datos.filtrados = datos[ , -altaCorrelacion]

# Transformacion
completos <- ncc(datos.filtrados)
incompletos <- nic(datos.filtrados)
cat("Datos completos: ",completos, " e incompletos: ",incompletos,"\n")

imputados <- mice(datos.filtrados)
datos.imputados <- complete(imputados)

# Generacion del modelo con SVM
library(e1071)
model = e1071::svm(C ~ ., data = datos.imputados, type = "C-classification")

# Generacion de la prediccion
pred = predict(model, test)

# Guardar el archivo para Kaggle
ids = 1:nrow(test)
res = data.frame(ids, pred)
colnames(res) = c("Id", "Prediction")

write.csv(res, file = "./datos/resultado.csv", row.names = FALSE, quote = FALSE)
