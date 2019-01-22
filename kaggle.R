library(dplyr)

# Carga de datos
datos = read.csv("./datos/train.csv", na.strings = c(" ", "NA", "?"))
test = read.csv("./datos/test.csv", na.strings = c(" ", "NA", "?"))

datos = as.tibble(datos)
test = as.tibble(test)

library(Hmisc)
Hmisc::describe(datos)

library(ggplot2)
ggplot(datos) + 
  geom_bar(mapping = aes(x = C, y = ..prop.., group = 1))

resumen = dplyr::group_by(datos, C) %>%
  dplyr::summarise(nc = n())

# Datos perdidos
library(mice)

patron = mice::md.pattern(x = datos[1:200, 1:20], plot = TRUE)

completas = mice::ncc(datos)
incompletas = mice::nic(datos)

incompletas = mice:nic(test)

# VIM y Amelia para obtener el patron de datos perdidos
library(Amelia)
library(lattice)

Amelia::missmap(datos[1:50, 1:20], col=c("red", "steelblue"))

# Ordenar las variables por datos perdidos
porOrden = sort(sapply(datos, 
  function(x) {
    sum(is.na(x))
  }
))

porOrden

# Imputacion de valores perdidos
imputados = mice::mice(datos, m = 1, method = "cart", printFlag = TRUE)
incompletos = mice::nic(imputados) # Hay un error por las variables correladas, se verá más adelante

imputados = Amelia::amelia(datos, m = 1, parallel = "multicore", noms = "C")
incompletos = mice:nic(imputados$imputations$imp1) # Algo no ha funcionado aqui

library(corrplot)
corrMatrix = cor(na.omit(datos))
corrplot::corrplot(corrMatrix, 
                   order = "FPC", 
                   type = "upper", 
                   tl.col = "black", 
                   tl.str = 45) # FPC ordena de izquierda a derecha las mas correladas

### Este grafico es potente, pero muy costoso
### library(PerformanceAnalytics)
### PerformanceAnalytics::chart.Correlation(na.omit(datos[ , 1:5]), histogram = TRUE)

# Filtrado de variables
altaCorrelacion = caret::findCorrelation(corrMatrix, cutoff = 0.8)
filtrado = datos[ , -altaCorrelacion]
filtrado = as.tibble(filtrado)

# Modelo con OneR
library(OneR)
modelo1R = OneR::OneR(C ~ ., data = datos)
summary(modeloR)

modelo1R = OneR::OneR(C ~ ., data = filtrado)

filtradoDiscretizado = OneR::optbin(as.data.frame(filtrado))
modelo2R = OneR::OneR(C ~ ., data = filtradoDiscretizado)

# Generacion de la prediccion
prediccion = predict(modelo2R, as.data.frame(test))
prediccion = as.vector(prediccion)

# Guardar el archivo para Kaggle
ids = 1:nrow(test)
res = as.data.frame(cbind(ids, prediccion))
colnames(res) = c("Id", "Prediction")

write.csv(res, file = "./datos/resultado.csv", row.names = FALSE, quote = FALSE)

# JRipper
library(RWeka)

# Carga de datos
datos = read.csv("./datos/train.csv", na.strings = c(" ", "NA", "?"))
test = read.csv("./datos/test.csv", na.strings = c(" ", "NA", "?"))

datos[ , "C"] = as.factor(datos[ , "C"])
modeloJR = RWeka::JRip(C ~ ., data = datos) # JRipper no puede leer valores numericos

RWeka::evaluate_Weka_classifier(modeloJR, numFolds = 10)

# Arboles de clasificacion
modeloJ48 = RWeka::J48(C ~ ., data = datos)
RWeka::evaluate_Weka_classifier(modeloJ48, numFolds = 10)

# Facilidades de prueba para eleccion de parametros
library(caret)
library(mlbench)
library(gbm)
data(Sonar)

enTrain = caret::createDataPartition(Sonar$Class, p = .75, list = FALSE)
train = Sonar[enTrain, ]
test = Sonar[-enTrain]

fitControl = caret::trainControl(method = "repeatedcv",
                                 number = 5,
                                 repeats = 3)
modelo = train(Class ~ ., data = train, method = "gbm",
               trControl = fitControl, verbose = FALSE)

grid = expand.grid(interaction.depth = c(1, 5, 9),
                   n.trees = c(1:10) * 50,
                   shrinkage = 0.1,
                   n.minobsinmode = 20)

modelo2 = train(Class ~ .,
                data = train,
                method = "gbm",
                trControl = fitControl,
                verbose = FALSE,
                tuneGrid = grid) # ERROR

# Paquete de referencia para SVM: e1071

