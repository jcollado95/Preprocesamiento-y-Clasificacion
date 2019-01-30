# Carga de datos
train = read.csv("./datos/train.csv", na.strings = c(" ", "NA", "?"))
test = read.csv("./datos/test.csv", na.strings = c(" ", "NA", "?"))

train$C = factor(train$C)

IQR <- sapply(na.omit(train[ , -51]), IQR)
BT <- sapply(na.omit(train[ , -51]), quantile, probs = .25) - 100 * IQR
TT <- sapply(na.omit(train[ , -51]), quantile, probs = .75) + 100 * IQR

# Eliminacion de outliers
trainSinClase <- train[ , -51]
outliers <- trainSinClase

for ( i in c(1:ncol(trainSinClase))) {
  outliers[,i] <- (trainSinClase[,i] > BT[i] & trainSinClase[,i] < TT[i])
}

outliers[is.na(outliers)] <- TRUE
outliersFilas <- as.logical(apply(outliers,1,prod))
dataLimpio <- trainSinClase[outliersFilas,]

dataTrain <- cbind(dataLimpio,"C"=train[rownames(dataLimpio),51])

# Imputacion de valores perdidos
library(mice)
imputados <- mice(dataTrain, m = 5, method = "pmm")
dataTrain <- complete(imputados)

# Filtrado de variables muy correladas
library(caret)
library(corrplot)
corrMatrix = cor(na.omit(dataTrain[,-51]))
corrplot::corrplot(corrMatrix, 
                   order = "FPC", 
                   type = "upper", 
                   tl.col = "black", 
                   tl.str = 45) # FPC ordena de izquierda a derecha las mas correladas

altaCorrelacion = findCorrelation(corrMatrix, cutoff = 0.8)
dataTrain = dataTrain[ , -altaCorrelacion]

# CHECK: Normalizacion de los datos
library(caret)
preprocesamiento <- preProcess(dataTrain[,1:(ncol(dataTrain)-1)],method=c("center","scale"))
dataTrain[ , -ncol(dataTrain)] <- predict(preprocesamiento,dataTrain[,1:(ncol(dataTrain)-1)])

# TO DO: Tune best parameters (C and Gamma) -> tune.svm()


# Generacion del modelo con SVM
library(e1071)
tuned_parameters <- tune.svm(C~., data = dataTrain, gamma = 2^(-1:-1), cost = 2^(-2:2))
summary(tuned_parameters)
plot(tuned_parameters)

model = e1071::svm(C ~ ., data = dataTrain, type = "C-classification")

# Preprocesamiento test
test = test[ , -altaCorrelacion]

IQR <- sapply(test,IQR)
Q1 <- sapply(test,quantile,probs=.25)
Q3 <- sapply(test,quantile,probs=.75)
TT <- Q3 + 100*IQR
BT <- Q1 - 100*IQR

outliers <- test

for ( i in c(1:ncol(test))) {
  outliers[,i] <- (test[,i] > BT[i] & test[,i] < TT[i])
}

outliersFilas <- as.logical(apply(outliers,1,prod))
medias <- apply(test[outliersFilas,], 2, mean)

dataTest <- test
for (x in c(1:ncol(test))) {
  dataTest[!outliersFilas,x] <- medias[x]
}

test <- predict(preprocesamiento, dataTest)

# Generacion de la prediccion
pred = predict(model, test)

# Guarda el archivo para Kaggle
ids = 1:nrow(test)
res = data.frame(ids, pred)
colnames(res) = c("Id", "Prediction")

write.csv(res, file = "./datos/resultado.csv", row.names = FALSE, quote = FALSE)
