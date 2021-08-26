# Permutation invariance and equivariance
## Permutation Invariance
Invariância de permutação significa que a função não depende da ordem das linhas / colunas na matriz de adjacência

## Permutation equivariance
  A equivariância de permutação significa que a saída de f é permutada de forma consistente quando permutamos a matriz de adjacência

# VISÃO GERAL DA ESTRUTURA DE PASSAGEM DE MENSAGEM
Seja um grafo $G = (V,E)$ juntamente com um conjunto de features dos nós $X \in R^{d x |V|}$ 
Durante cada iteração de passagem de mensagem em um GNN, uma hidden embedding $h_u^{(k)}$, para a cada nó $u \in V$, é atualizado de acordo com as informações agregadas da vizinhança do grafo $N(u)$. Esta atualização de transmissão de mensagens pode ser expressa da seguinte forma:

<img src="https://render.githubusercontent.com/render/math?math=\large h_u^{(k+1)} = UPDATE^{(k)} \left ( h_u^{(k)}, AGGREGATE^{(k)}(\{h_v^{(k)}, \forall v \in N(u)\} \right )">

$$\large h_u^{(k+1)} = UPDATE^{(k)} \left ( h_u^{(k)}, AGGREGATE^{(k)}(\{h_v^{(k)}, \forall v \in N(u)\} \right )$$
$$\large h_u^{(k+1)} = UPDATE^{(k)} \left ( h_u^{(k)}, m^{(k)}_{N(u)} \right )$$

* UPDATE e AGGREGATE são duas funções diferenciáveis
*  $\large m_{N(u)}$ é a "mensagem" que é agregada da vizinhança de u do grafo $N(u)$

A cada iteração k do GNN, a função agregada toma como entrada o conjunto de informações dos nós na vizinhança de u no grafo $N(u)$ e gera uma mensagem $m^{(k)}_{N(u)}$ baseado nesta informação agregada da vizinhança. A atualização da função de atualização então combina a  mensagem $m^{(k)}_{N(u)}$ com a incorporação anterior $h_u^{(k-1)}$ do nó u para gerar a incorporação atualizada $h_u^{(k)}$ . Os embeddings iniciais em $k = 0$ são definidos para as features de entrada para todos os nós, ou seja, $\large h_u^{(0)} = x_u, \, \forall u \in V$. 

Depois de executar K iterações de passagem de mensagem do GNN, podemos usar a saída da camada final para definir os embeddings para cada nó, ou seja,
$$\large z_u = h_u^{(k)}, \, \forall u \in V$$

![[Pasted image 20210809115658.png]]
Esta visualização mostra uma versão de duas camadas de um modelo de passagem de mensagens

https://miro.medium.com/max/700/1*oSQyFjtUkI7_u7lJXWU68Q.gif

## Motivacao e intuicao
A intuição basica por trás da GNN message-passing é agregar a informação da vizinhança a cada interação, quanto mais iterações mais informação é obtida.
Para k = 1, cada nó possui informação de 1-salto vizinho, generalizando, para k iterações o nó oculto terá informação de k-salto vizinho

Mas qual informação está sendo codificada nessas unidades ?
* Por um lado temos informações estruturais sobre o grafo, por exemplo, após k iterações, a unidade $h_u^{(k)}$ do nó u pode codificar informação sobre o grau de todos os nós na vizinhança a uma distancia de k-saltos, essa informação estrutural pode ser util para analizar estruturas quimicas
* Outra informação capturada pela nossa unidade é sobre as features a uma distancia de k-saltos

 ## A GNN basica
 Da definição anterior:
 $$ m^{(k)}_{N(u)}  = \sum_{v \in N(u)} h_v $$
 $$ UPDATE^{(k)} \left ( h_u^{(k)}, m^{(k)}_{N(u)} \right ) = \sigma \left ( W^{(k)}_{self} \, h_u^{(k-1)} + W^{(k)}_{neigh} \, m^{(k)}_{N(u)} \right )$$
 
 A GNN básica de passagem de mensagens é definida como  
 $$\large h_u^{(k)} = \sigma \left ( W^{(k)}_{self} \, h_u^{(k-1)} + W^{(k)}_{neigh} \, \sum_{v \in N(u)} h_v^{(k-1)} + b^{(k)} \right ) $$
 
 Onde $W^{(k)}_{self}, W^{(k)}_{neigh} \in R^{d(k) \, X \, (k-1)}$

## Equação a nível do Grafo
$$ H^{(t)} = \sigma \left ( AH^{(k-1)}W^{(k)}_{neigh} + H^{(k-1)}W^{(k)}_{self} \right )$$

Onde $H^{(k)} \in R^{|V| \,\, X \,\, d}$
Cada nó corresponde a uma linha da matriz
 ## Passagem de mensagem com self-loop
 
 $$\large h_u^{k} = AGGREGATE(\{h_v^{(k - 1)}, \forall v \in N(u) \, \cup \, \{ u \} \} ) $$
 
 Temos dessa forma que o a função update está implicitamente definida no metódo de agregação
 
 $$ H^{(t)} = \sigma \left ( (A+I) H^{(t-1)}W^{t} \right )$$
 Adicionando a matriz de identidade adicionamos o self-loop
 
# Generalização da agregação dos vizinhos
Até agora estamos somando todos os vizinhos, entretanto, existem outras maneiras mais eficientes para agregar os nós vizinhos e para a operação update

## Normalização dos vizinhos
Temos um problema na abordagem de somar os vizinhos, quando temos um grande número de vizinhos (um nó com grau muito alto) teremos instabilidade númerica, assim como dificuldades para otimizar
Uma solução para esse problema é normalizar a agregação baseado no grau do nó envolvido:
$$\large m_{N(u)} = \frac{\sum_{v \, \in \, N(u)} h_v}{|N(u)|} $$

### Normalização simetrica
$$\large m_{N(u)} = \sum_{v \, \in \, N(u)} \frac{ h_v}{\sqrt{|N(u)| |N(u)|}} $$

# GCNs
Uma dos modelos mais populares - Graph Convolutional network - emprega a normalização simetrica e a abordagem do self-loop update.  O modelo GCN, portanto, define a função de passagem de mensagens como:

$$\large h_u^{(k)} = \sigma \left ( W^{(k)} \,  \, \sum_{v \, \in \, N(u) \, \cup \, \{u \} } \frac{ h_v}{\sqrt{|N(u)| |N(u)|}}  \right ) $$

Uma camada GCN básica é definida no Kipf e Welling [2016a] como:

$$\large H^{(k)} = \sigma \left (  \tilde{A}H^{(k-1)}W^{(k)}\right )$$

(matriz laplaciana) [Matriz Laplaciana – Wikipédia, a enciclopédia livre (wikipedia.org)](https://pt.wikipedia.org/wiki/Matriz_Laplaciana)

Onde $\tilde{A} = (D+I)^{-1/2}(I+A)(D+I)^{-1/2}$ é uma variante normalizada da matriz de adjacencia (com self-loop) e $W^{(k)}$ é um parametro de pesos do aprendizado
