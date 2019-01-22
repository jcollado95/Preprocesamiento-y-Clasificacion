# Carga de datos
datos = read.csv("./datos/train.csv", na.strings = c(" ", "NA", "?"))
test = read.csv("./datos/test.csv", na.strings = c(" ", "NA", "?"))

# Filtrado de variables muy correladas
library(corrplot)
corrMatrix = cor(na.omit(datos))
corrplot::corrplot(corrMatrix, 
                   order = "FPC", 
                   type = "upper", 
                   tl.col = "black", 
                   tl.str = 45) # FPC ordena de izquierda a derecha las mas correladas

altaCorrelacion = caret::findCorrelation(corrMatrix, cutoff = 0.8)
datos.filtrados = datos[ , -altaCorrelacion]

# Generacion del modelo con SVM
library(e1071)
model = e1071::svm(C ~ ., data = datos.filtrados, type = "C-classification")
model2 = obj$best.model

# Generacion de la prediccion
pred = predict(model, test)
pred2 = predict(model2, test)

# Guardar el archivo para Kaggle
ids = 1:nrow(test)
res = data.frame(ids, pred2)
colnames(res) = c("Id", "Prediction")

write.csv(res, file = "./datos/resultado2.csv", row.names = FALSE, quote = FALSE)
