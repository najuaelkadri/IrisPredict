#Importando
library(caret)
library(rpart)
library(rpart.plot)

#Baixando os dados
data("iris")

#Dar uma olhada na estrura e nos dados
str(iris)
summary(iris)

#Vou dividir o conjunto de dados em input (X) e output (y).
#Nesse caso, os inputs são as quatro dimensões (comprimento da sépala, largura da sépala,
#comprimento da pétala e largura da pétala) e o output é a espécie da planta.
X <- iris[, 1:4]   #O espaço em branco antes da vírgula indica que quero selecionar todas as linhas do conjunto de dados.
                   #1:4 especifica as colunas a serem selecionadas, que são as colunas 1, 2, 3 e 4 do conjunto de dados da íris.
                   #As quatro dimensões (comprimento da sépala, largura da sépala, comprimento da pétala e largura da pétala) são representadas por essas quatro colunas no conjunto de dados da íris
y <- iris$Species

#Para avaliar o desempenho do modelo, vou dividir os dados em treinamento e teste.
#O treinamento será usado para treinar o modelo e o conjunto de teste será
#usado para avaliar sua precisão.Descobri que se usa o pacote caret para dividir os dados.

set.seed(123) #Aqui, defini uma semente aleatória para garantir a reprodutibilidade dos resultados. 
              #Definindo isso, uma semente nos permite obter as mesmas divisões aleatórias toda vez que o código é executado.

train_index <- createDataPartition(y, p = 0.7, list = FALSE) #A variável y representa o output.
                                                             #O argumento p é definido como 0,7,indicando que quero 70% dos dados para treinamento.

#Separando os dados em dois conjuntos: X_train e y_train para treinar o modelo e X_test e y_test para avaliar o desempenho do modelo em dados não vistos. 
#O conjunto de treinamento é usado para construir o modelo, enquanto o conjunto de teste é usado para avaliar o quão bem o modelo generaliza para dados novos e não vistos.
X_train <- X[train_index, ]  #"train_index" contém as observações que serão utilizadas para treinar o modelo.
                             #X[train_index, ] seleciona as linhas do conjunto de dados X correspondentes aos índices em train_index.
                             #Isso cria um subconjunto do conjunto de dados original X, contendo apenas as observações que serão usadas para treinar o modelo.
                             #O subconjunto é atribuído à variável X_train, representando os recursos de treinamento.
y_train <- y[train_index]    

X_test <- X[-train_index, ]  #[-train_index] seleciona as linhas do conjunto de dados X que não estão incluídas no train_index.
                             #Isso cria um subconjunto do conjunto de dados original X, contendo as observações que serão usadas para testar o modelo.
                             #O subconjunto é atribuído à variável X_test, representando os recursos de teste.
y_test <- y[-train_index]    

#A função rpart() do pacote rpart foi usada para treinar o modelo de árvore de decisão. 
#Ele usa os dados de treinamento (X_train e y_train) e a fórmula y_train ~ . para especificar
#que a espécie (y_train) deve ser prevista com base em todas as outras variáveis (comprimento da
#sépala, largura da sépala, comprimento da pétala e largura da pétala).
model <- rpart(y_train ~ ., data = X_train) 

rpart.plot(model) #minha árvore que lindinha

#A função predict() é usada para fazer previsões nos dados de teste (X_test) usando o modelo treinado.
#O argumento "type = "class" especifica que os valores previstos devem ser as classes.
predictions <- predict(model, X_test, type = "class")

#A função confusionMatrix() do pacote caret é usada para calcular a confusion matrix e outras
#métricas de desempenho. Ele usa os valores previstos (previsões) e os rótulos reais (y_test) como input.
confusion_matrix <- confusionMatrix(predictions, y_test)

#A precisão é extraída do objeto confusion_matrix usando o componente $overall e o índice 'Accuracy'.
accuracy <- confusion_matrix$overall['Accuracy']

print(confusion_matrix)
print(accuracy)

# Convertendo a confusion matrix para data frame
confusion_df <- as.data.frame(confusion_matrix$table)

#gráfico de gradiente de calor
# Vou renomear as coluna no data frame
colnames(confusion_df) <- c("Referência", "Predição", "n")

# Plot the confusion matrix as a heatmap
ggplot(confusion_df, aes(x = Referência, y = Predição, fill = n)) +
  geom_tile() +                                                      #A função geom_tile() cria blocos coloridos
                                                                     #representando os valores na confuion matrix.
  geom_text(aes(label = n), color = "white") +                       #A função geom_text() adiciona nome aos blocos.
  scale_fill_gradient(low = "lightblue", high = "darkblue") +        #A função scale_fill_gradient() define o gradiente de cores.
  labs(x = "Referência", y = "Predição", title = "Confusion Matrix") #A função labs() define os nome no gráfico.

#gráfico de barras #ficou estranho vou arrumar depois
colnames(confusion_df) <- c("Referência", "Predição", "Contagem")

ggplot(confusion_df, aes(x = Referência, y = Contagem, fill = Predição)) +
  geom_bar(stat = "identity", position = "dodge") +                     #A função geom_bar() cria um gráfico de barras
                                                                        #com os valores previstos (Previsão) como cores 
                                                                        #de preenchimento.
  labs(x = "Referência", y = "Contagem", title = "Confusion Matrix") +  #O argumento "position = "dodge"" posiciona as 
                                                                        #barras lado a lado.
  theme(axis.text.x = element_text(angle = 45, hjust = 1))              #A função theme() é usada para personalizar a 
                                                                        #aparência do gráfico, girando os rótulos do eixo x.
