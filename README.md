# hiaac-experiment-executor

Experiment Executor Framework v0.0.1. 


## Usando o executor de experimentos (tutorial)
Leia a documentação do executor de experimentos (README.md) (mais informações dentro de `experiments/experiment_executor`) e tente aprender o uso dos arquivos YAML e como executar os exemplos. Execute os experimentos no terminal, algo mais ou menos semelhante ao seguinte comando:

```
python execute.py examples/experiment_configurations/ -o examples/results/ -d ~/work/datasets/
```

Este comando irá executar os experimentos descritos dentro do diretório `examples/experiment_configurations` e colocará os resultados para cada experimento em `examples/results`. O parâmetro `-d` incida a raiz do diretório com os conjuntos de dados (dentro desta raiz deve haver uma pasta `standartized_balanced` e dentro dela subdiretórios para cada conjunto de dados, e.g. KuHar)

Tente criar um arquivo de configuração que:
1. Use  dataset Kuhar standartized_balanced (treino: train/validation e teste: test)
2- Execute um outro classificador do scikit-learn, sem ser o RanfomForest, SVM ou KNN para fazer a classificação. Uma lista de classificadores pode ser achada no site do sklearn. Para isso você deve editar o arquivo `config.py`, especialmente a variável `estimator_cls` (que é um dicionário) e adicionar um classificador novo. Ao adicioná-lo aqui, ele fica visível para os arquivos de configurações YAML pelo nome da chave do dicionário
3. Não use nenhum redutor de dimensionalidade
4. Não use nenhum escalador
5. Transforme os dados para o domínio da frequência
6. Use todas as features

**Nota**:  Os parâmetros para a construção do modelo (para a função `__init__` do seu classificador) é passado pela opção `kwargs`, na seção estimators do seu classificador do arquivo de configuração YAML.

## Adição de novos classificadores
Você pode adicionar outros classificadores feitos por si (e.g, rede neural) ao executor facilmente. Todo classificador novo deve implementar a API do scikit learn de `fit/predict`, com a seguinte assinatura (deve ser exatamente esta):

* `fit(self, X,y=None)`: onde X é o dado (numpy array) e y o label (caso haja). Esta função deve treinar o seu modelo com estes dados. O retorno é o próprio objeto (retorna self).
* `predict(self, X)`: onde X é um dado (numpy array). Esta função deve redizer os rótulos e retornar o vetor de elementos preditos.

Por exemplo, vamos criar um arquivo `models.py` no `experiment_executor` e vamos colocar o seguinte código nele, de uma rede ficticia.

```python
from librep.base.estimator import Estimator
import torch
from torch.utils.data import TensorDataset, DataLoader

# Códido da minha rede Pytorch
class MinhaRede(torch.nn.Module):
    def __init__(self):
        self.layer1 = torch.nn.Conv1d(in_channels=7, out_channels=20, kernel_size=5, stride=2)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Conv1d(in_channels=20, out_channels=10, kernel_size=1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        log_probs = torch.nn.functional.log_softmax(x, dim=1)
        return log_probs




# Código do meu classificador (que instancia MinhaRede)
class MeuClassificador(Estimator):
	# Cada classificador define seu próprio init, com as
# suas informações pertinentes. Essas são de exemplo
def __init__(self, epochs=10, learning_rate=1e-4):
		self.model = MinhaRede()
		self.epochs = epochs
		self.lr = learning_rate

	# Treina 
	def fit(self, X, y):
		# Converte X, y de numpy para tensores
tensor_x = torch.Tensor(X)
tensor_y = torch.Tensor(y)
# Cria um dataset torch
my_dataset = TensorDataset(tensor_x,tensor_y)
# pode criar um de validação aqui também
my_dataloader = DataLoader(my_dataset)
		# itera as épocas
for i in self.epochs:
for data, label in my_dataloader:
... laço de treinamento de self.model ...
return self

def predict(self, X):
	... retorna a predição de X de self.model ...
```

Uma vez adicionado, vá no arquivo config.py e adicione uma entrada em `estimator_cls` para este modelo.  Um exemplo da alteração da variavável estimator_cls para este caso seria:

```python
...
from models import MinhaRede
...
estimator_cls = {
	"SVM": SVC,
	"KNN": KNeighborsClassifier,
	"RandomForest": RandomForestClassifier,
	“MinhaCNN”: MinhaRede
}
...
```

Crie um arquivo de configuração com este modelo e execute.
