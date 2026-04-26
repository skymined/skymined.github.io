---
tags:
  - paper/RAG
created: 2026-03-01T14:32:00
---


![](https://velog.velcdn.com/images/adsky0309/post/37b89fd6-5582-40c7-b66d-bfb6980d7ec0/image.png)

논문 리뷰를 시작하기 전, 해당 논문을 이해하기 위해 찾아봤던 개념들을 먼저 서술해보고자 한다. 

## 0 들어가기 전
### ODQA란?
Open domain Question Answer의 약자로, 특정 도메인에 국한을 두지 않고 질문이 들어왔을 때 DB에 관련 문서를 찾아 정답을 찾아R주는 것을 말한다.
ODQA의 논문 흐름을 보면 다음과 같다. (출처 : 고려대학교 산업경영공학부 DSBM 연구실 Youtube)

![](https://velog.velcdn.com/images/adsky0309/post/17101e46-3847-4f34-a5f3-0f363b418827/image.png) 

DrQA는 처음으로 Open Domain QA를 제안한 모델로 Retriever Reader Model을 사용한다. DrQA의 성능을 보안한 것이 ORQA로 해당 모델은 Retriever를 학습시키고자 하였다. 이후 REALM은 Retriever 뿐만 아니라 Reader도 함께 학습시킨 모델이다.

이후 RAG가 등장하는데, ODQA와 KIT(Knowledge Intensive Task)를 이어주는 모델로 Retriever 뿐만 아니라 Generator(생성기)도 함꼐 학습시킨 것이다.

각 모델에 대한 논문이 있는데, 우선은 이러한 흐름으로 진행된다는 것을 알면 좋을 것 같다.

### BM25 vs DPR
**BM25**는 주어진 Query에 대해 문서와의 연관성을 평가하는 랭킹함수로 사용되는 알고리즘이다. TF-IDF 계열의 검색 알고리즘의 하나인데, 같은 단어를 다르게 표현할 경우 성능이 떨어진다는 단점이 있다.
예를 들어 Vilian이라는 단어가 있다면, 이를 다른 방식으로 표현할 수 있을 것이다.
이러한 점을 개선한 것이 **DPR, Dense Passage Retriever**이다.

DPR은 모든 Passage에 대한 index를 연속적인 저차원 공간으로 dense하게 매핑하는데, 이때 동의어 등이 query와 passage 안에 없더라도 숨어있는 의미를 발견하여 passage를 고를 수 있다. 

<br>

## 1 Introduction
Pre-train(사전 학습)된 neural language model은 memory를 확장하거나 개정할 수 없고, 예측을 통한 통찰을 제공할 수 없으며 hallucination 현상을 일으키기도 한다.

이러한 단점 때문에 parametric memory를 non-parametric memory와 결합한 모델을 만들어 이런 문제를 해결하기도 하였다.
REALM과 ORQA는 masked language model과 retriever를 결합한 모델은 좋은 성능을 보였으나 이전에는 오픈 도메인 질문 답변 테스크만을 수행할 수 있었다.

따라서 이의 연장선으로 본 논문은 RAG라 부르는 general-purpose fine-tuning을 통해 사전학습된 parametric-memory 생성 모델에 non-parametric memory를 부여한 모델을 만들었다.

이때 parametric memory는 사전학습된 seq2seq transformer이고 non-parametric memory는 dense vecor index of Wikipedia이다.

## 2 Methods
![](https://velog.velcdn.com/images/adsky0309/post/e4bc42bd-6658-4e82-baf7-f41eea5cf24f/image.png)

여기서 우리가 필요한 것은 **1) pretrained generator model, 2) pretrained Retriever model, 3) Indexed knowledge base(KB)**이다.

RAG는 input sentence인 x가 들어오게 되면 non-parametric 메모리에서 관련된 문서인 z를 찾고, 이를 활용하여 target sentence인 y를 생성한다.
이때 q(x)는 query encoder를 통해서 나온 representation이다. z는 문서 덩어리로 우리가 사용하는 Wikipedia를 100단어씩 끊어놓은 한 문장을 말한다. d(z)는 문서 z 의 representation을 말한다.)

RAG는 크게 Retriever와 Generator 두 개의 요소로 구성되어 있는데 Retriever는 x에서 z를 찾는 담당이라면($p_n(z|x)$는 x가 주어졌을 때 z가 나올 확률) Generator($p_\theta$)는 input sequence x와 documents z, 그리고 1부터 i-1까지 token(y)가 주어졌을 때 $$y_i$$가 나올 확률을 계산하며 생성을 진행한다.

Retriever과 Generator를 end-to-end로 훈련시키기 위해서 retrieved document, z를 latent variable로 다루었다.

또한 두 가지 모델을 제안하였는데 하나는 동일한 문서를 사용하여 다음에 올 토큰을 예측한 RAG-Sequence이고, 다른 하나는 여러 문서를 이용하여 다음에 올 토큰을 예측하는 RAG-Token이다.
<br>


아래는 논문이 사용한 모델(RAG-Sequence, RAG-Token), Retriever(DPR), Generator(BART)에 대한 설명이다.


### 2.1 Models

#### RAG-Sequence Model

![](https://velog.velcdn.com/images/adsky0309/post/ae239d65-f35d-4d8f-b31e-bb3b0bf38771/image.png)


해당 수식을 분해해보자면 다음과 같다.

먼저 Retrieval 단계, $p_\eta(z|x)$ 에서는 주어진 입력 x에 대해 가장 관련 있는 문서 z가 나올 확률이다.

Generation 단계 $p_\theta(y|x,z)$는 입력 x와 검색된 문서 z를 바탕으로 응답 y가 나올 확률이다.

이 두 모델을 결합하여 RAG 모델은 전체적인 생성과정을 end-to-end 방식으로 학습할 수 있다. 두 번째 식은 위의 식을 i번째 단어로 표현한 것이다.

따라서 RAG-Sequence 모델은 각각의 문서를 이용해 output sequence 전체를 대상으로 값을 산출하고, 문서에 대해 marginalize 하여 최종값을 산출하는 것이다.

#### RAG-Token Model

![](https://velog.velcdn.com/images/adsky0309/post/674046d3-6cf6-4029-957d-5f92323e2fa3/image.png)


RAG-Token의 경우, 각 토큰 $y_i$에 대해 문서 z의 확률을 계산하고 이를 합산한다.

RAG-Sequence 모델이 문서 z를 먼저 선택한 후 전체 문장 y를 한 번에 생성하는 것과 비교하면, 각 토큰 $$y_i$$를 생성할 때마다 문서 z를 선택하여 확률을 계산한다.

때문에 RAG-Token 모델은 각 토큰을 생성할 때마다 문서 z를 고려하여 보다 세밀하게 응답을 생성할 수 있다.

> - **RAG-Sequence**는 문서에 대한 값을 sequence 단위로 고려한 다음 marginalize하는 것
- **RAG-Token**은 문서에 대한 값을 token 단위로 고려한 다음, marginalize 하고, 그 다음 token을 생성하면서 sequence를 생성하는 것이다.

### 2.2 Retriever : DPR

검색 구성요소 $p_\eta(z|x)$ 는 DPR을 기준으로 하며 DPR는 bi-encoder 아키텍쳐를 사용한다.

![](https://velog.velcdn.com/images/adsky0309/post/ca8f557a-2c58-4478-b586-7b874343f189/image.png)


$d(z)$는 $BERT_{BASE}$로 생성한  문서의 dense representation이고 $q(x)$는 다른 매개변수를 가진 $BERT_{BASE}$ transformer에 의해 생성된 query representation이다.

가장 높은 가능성이 있는 k개의 요소 z의 리스트 top-k($p_\eta(\cdot|x)$)을 구하는 것은 MIPS(Maximum Inner Product Search) 문제이며 sub-linear time을 푸는 것으로 근사할 수 있다.

DPR로부터 사전학습된 bi-encoder를 사용하여 리스티버를 초기화하고 문서 인덱스를 만든다.

학습된 리스티버는 TriviaQA Question과 Natural Question을 통해 정답이 포함된 문서를 검색할 것이다. 여기서 문서 인덱스(document index)는 non-parametric memory이다.

### 2.3 Generator : BART

생성기 $p_\theta(y_i|x,z,y_{1:i-1})$는 임의의 encoder-decoder로 구현이 가능하다. 논문에서는 400M의 파라미터를 가지고 있는 사전학습된 seq2seq transfomer 모델인 **BART-large**를 사용하였다.

입력 x와 검색된 내용 z를 결합하기 위해 단순히 concat하였다.

### 2.4 Training

정확히 무엇을 찾아야 하는지에 대한 직접적인 학습 없이 DPR 기반의 리트리버와 BART-LARGE 기반의 생성기(Generator)를 동시에 훈련시켰다. input/output pairs, 파인 튜닝에 사용되는 학습 말뭉치가 주어지면 각 타겟에 대한 negative marginal log-likelihood, $\sum_j-logp(y_j|x_j)$를 최소화하는 방식으로 학습된다.

문서 인코더 $BERT_d$를 업데이트하기 위해서는 비용이 많이 들고 document index는 주기적으로 업데이트가 되어야 한다. 이는 강한 성능에 꼭 필요한 요소가 아니기 때문에 document encoder와 index는 고정하고, query encoder인 $BERT_q$와 $BART$ 생성기만을 파인튜닝하였다.

### 2.5 Decoding

테스트에서 RAG-Sequence와 RAG-Token은 다른 방식으로 $arg  max_yp(y|x)$를 게산하였다.

#### RAG-Token

RAG-Token 모델은 아래와 같은 전이 확률( transition probability)를 가지는 일반적인 자기회귀적 seq2seq 모델이다. 

$$
p_\theta(y_i|x, Y_{1:i-1})=\sum_{z\in top-k(p(\cdot|x))}p_\eta(z_i|x)p_\theta(y_i|x, z_i, y_{1:i-1})
$$

위의 식은 RAG-Token 모델의 전이 확률을 나타낸다. 특정 시점 i에서 출력 토큰 $y_i$가 입력 x와 이전 출력 시퀀스 $Y_{1:i-1}$를 기반으로 생성될 확률을 나타낸다.

이때 **_검색된 상위 k개의 결과를 고려하여 각 검색결과 z에 대해 생성 확률을 계산하고, 이들을 합산하여 최종적으로 $y_i$가 생성될 확률을 계산_**하는 것이다.

디코더 과정에서는 $p_\theta(y_i|x, Y_{1:i-1})$를 standard beam decoder에 넣어 구한다.

#### RAG-Sequence

가능도 $p(y|x)$은 이전에 사용해 왔던 토큰 당 가능도로 분해할 수 없기 때문에 빔 서치(beam search)로 해결할 수 없다. 따라서 각 문서 z에 대해 beam search를 진행하고 $p_\theta(y_i|x,z,y_{1:i-1})$를 이용해 각 가정의 점수를 측정한다. 이에 가설 집합 Y가 만들어지는데 이중 일부는 어떤 문서에서도 보이지 않을 수 있다.

가설 y에 대한 가능도를 측정하기 위해 y가 빔에 나타나지 않은 문서 z에 대해 추가적을 순방향 계산을 거치고, 생성기의 확률 $p_\eta(z|x)$을 곱한 다음 빔 전체에 걸친 확률을 더해 marginals를 계산해야 한다. 이 디코딩 과정을 ‘**Thorough Decoding**’이라고 한다.

더 긴 출력 시퀀스를 위해서는 $|Y|$의 값이 커지고 더 많은 순방향 계산이 필요하다. 더 효과적인 디코딩을 위해 x, z로부터 빔 서칭되는 동안 y가 생성되지 않은  $p_\theta (y|x, z_i) \approx 0$에 대한 근사값을 만들 수 있다. 이를 통해 후보 집합 Y가 만들어지면 추가적인 순방향 계산이 이뤄지는 것을 피할 수 있다. 이를 ‘**Fast Decoding**’이라고 한다.

<br>


## 3 Experiments

연구진은 넓은 범위의 지식 집약적 태스크(knowledge-intensive task)에서 RAG를 실험하였다. 모든 실험에서는 non-parametric knowledge source로 단일 위키피디아 덤프를 사용하였다.

각 위키피디아 글은 100 단어로 분할되어 총 21M개의 문서를 생성했다. 문서 인코더를 사용하여 각각의 문서의 임베딩을 계산하였고 FAISS를 이용하여 하나의 MIPS를 구축했다.

학습과정에서는 각 쿼리에서 상위 k개의 문서를 검색하였고 k값은 $k\in ({5,10})$이다. (모델이 각 쿼리에 대해 상위 5개 또는 10개의 문서를 검색하여 사용했다는 뜻이다.)

### 3.1 Open-domain Question Answering

ODQA는 질문에 대해 주어진 텍스트에 국한되지 않고 대규모 텍스크 컬렉션(ex.위키피디아)을 활용하여 답변을 찾는 태스크이며 자주 사용되는 지식 집약적 태스크이다.

문제와 답의 쌍 (x,y)를 입출력 텍스트 쌍으로서 RAG를 훈련시켰고 RAG와 추출적 QA paradigm을 비교하였다. 추출된 QA는 검색된 문서에서 어떤 영역을 추출하여 정답으로 사용한다.

또한 리트리버를 이용하지 않고 학습된 지식만을 사용해 답을 생성하는 ‘Closed-Book QA’와 비교하였다.

데이터셋으로는 유명한 4개의 open-domain QA dataset을 사용하였다. (NQ, TQA, WQ, CT)

### 3.2 Abstractive Question Answering

AQA는 질문에 대한 답변을 원문에서 직접 추출하는 대신, 원문을 기반으로 새로운 표현을 생성하여 답변을 제공하는 태스크이다. 기계 번역과 유사하게 원문 내용을 요약하거나 재구성하여 답변을 만드는 것을 목표로 한다.

RAG의 자연어 생성 능력(Natural language generation, NLG)을 평가하기 위해 MSSMARCO NLP 태스크를 사용하였다. 해당 태스크는 문제와 검색한 10개의 정답 문서, 그리고 검색한 문서에서 가지고 온 정답 문장으로 이뤄진다. 추상적 QA 태스트를 위해 제공된 문서는 사용하지 않고 질문과 정답만을 사용하였다.

### 3.3 Jeopardy Question Generation

Jeopardy Question Generation은 퀴즈 쇼 ‘Jeopardy!’ 형식의 질문을 생성하는 태스크로 정답이 먼저 주어지면 이에 대한 질문을 생성하는 방식이다.

예를 들어 “The World Cup”이라는 Answer가 있다면 Jeopardy는 “In 1986 Mexico scored as the first country to host this international sports competition twice.”라는 Question을 생성하여야 한다.

non-QA 세팅에서의 RAG의 생성능력 평가를 위해 오픈 도메인 질문 생성에 대한 연구를 수행하였다.

### 3.4 Fact Verification(FEVER)

Fact Verification은 주어진 명제(사실 진술)이 참인지 거짓인지 판단하는 태스크이다. 해당 태스크는 사실 검정을 위해 외부 지식을 활용한다. 예를 들어 진시황제에 대한 문장이 있으면 여러 역사 문서들을 참고하여 그것의 참거짓 여부를 확인한다.

<br>

## 4 Results

### 4.1 Open-domain Question Answering

![](https://velog.velcdn.com/images/adsky0309/post/50f6adcd-f347-49ec-8205-c922fb1922c9/image.png)


Table1의 모델들은 다음과 같다. 우선 Closed Book은 모델이 외부 지식이나 문서 없이 학습된 지식을 가지고 질문에 답하는 방식이다. 이때 T-5-11B+SSM은 T5모델에 추가적인 사전 학습이 더해진 버전이다. Open Book은 외부 지식을 사용한 것인데 그중에서도 REALM과 DPR은 주로 Retriever, 검색 단계에 집중한 모델이다. 반면 RAG-Token과 RAG-Seq은 retrieval 뿐만 아니라 Generation 단계 역시 거친다.

RAG의 결과를 보면 TQA를 제외한 모든 ODQA 태스크에서 RAG는 최고 성능을 보인다. REALM과 T5+SSM과 달리 RAG는 비용이 드는 ‘Salient span masking’을 사용하지 않고도 높은 성능을 보인다. 


### 4.2 Abstractive Question Answering

![](https://velog.velcdn.com/images/adsky0309/post/74b4e2e7-07b8-47eb-8c48-49b69b610092/image.png)


MSMARCO NLP 태스크에서 RAG 모델은 BART보다 뛰어난 성능을 보인다.

![](https://velog.velcdn.com/images/adsky0309/post/c800c1ca-7395-46b2-ab80-e269c38ce29a/image.png)


위의 표는 RAG와 BART가 생성한 정답의 예시이다. RAG가 BART보다 사실에 근거한 답을 생성하며 환각이 덜 발생함을 알 수 있다.

### 4.3 Jeopardy Question Generation

위의 Table2를 보면 Jeopardy Question Generation 에서 RAG token이 RAG Sequence보다 성능이 좋음을 알 수 있다.

![](https://velog.velcdn.com/images/adsky0309/post/6955c4bf-2412-4df5-bb9e-8b7792a3a890/image.png)


Table 4는 RAG 와 BART의 출력을 사람이 비교한 결과이다. BART가 더 사실적인 경우는 7.1%인데 비해 RAG가 더 사실적인 경우는 42.7%에 달한다. 

![](https://velog.velcdn.com/images/adsky0309/post/d43ba709-4426-45d4-a7e6-8accfd293e5e/image.png)


해당 그림은 RAG-Token 모델이 특정 입력에 대해 Jeopardy 문제를 생성할 때 각 토큰에 대해 5개의 검색된 문서의 사후 확률을 어떻게 사용하는지 보여준다. 그림의 열은 각 토큰에 대해 5개의 문서가 선택된 후, 각 문서의 사후 확률을 나타내는 히트맵이고 각 셀은 특정 토큰을 생성할 때 문서가 얼마나 관련성이 높은지를 나타낸다.

히트맵을 보면 대부분 옅은 파란색으로 채워져 있고, 초기에는 특정 문서 (doc2)가 짙은 색상이지만 이 역시 확률이 넘차 낮고 균등해진다. 이는 모델이 특정 문서에 의존하지 않고 자체적으로 학습된 지식을 바탕으로 텍스트를 생성하고 있다고 해석할 수 있는 근거가 된다. (특정 문서만을 참고하지 않는다는 것)

### 4.4 Fact Verification

![](https://velog.velcdn.com/images/adsky0309/post/100d8f4e-54c0-4196-96d8-853cfc3aad5d/image.png)


FVR 역시 BART보다 높은 성능을 가지고 있음을 알 수 있다. SoTA 모델보다 뒤쳐지기는 하지만, SoTA 모델은 해당 도메인에 특화된 아키텍쳐를 포함하고 있고 추가적인 공학적 기술과 중간의 리트리버 학습이 이뤄졌다는 것을 감안해야 한다.

### 4.5 Additional Results

#### Generation Diversity

Jeopardy 태스크에서 RAG 모델이 BART 모델보다 사실 기반 생성 능력이 뛰어나다는 것을 알 수 있었다. 서로 다른 모델에서 생성 다양성을 평가하기 위해 전체 앵그램을 개별 앵그램으로 나눈 비율을 측정하였다.

![](https://velog.velcdn.com/images/adsky0309/post/0e688783-bf53-4bdd-ad3a-fd076b1414b0/image.png)


Table5은 RAG-Sequence가 RAG-Token보다 다양성이 높고 두 모델 모두  diversity-promoting decoding이 없어도 BART보다 높은 점수를 보임을 알 수 있다.

**Retrieval Ablations**

![](https://velog.velcdn.com/images/adsky0309/post/ece8527f-c92a-4aca-a76a-5079eb6703b8/image.png)


RAG-Token-BM25는 RAG의 Retriever인 DPR을 BM25로 변경한 것이고 RAG-Token-Frozen은 BART의 파라미터만 업데이트하고 Retriever는 파인튜닝하지 않은 것이다.

대부분의 결과에서 본래의 RAG가 더 나은 성능을 보였다. 다만 FVR의 경우 사실 확인 Task로 실제 문서에서 주요 토큰의 등장여부가 중요하기 때문에 주어진 쿼리에 대해 문서의 연관성을 평가하는 랭킹 함수로 사용되는 BM25가 더 성능이 좋게 나왔다고 판단할 수 있다.

#### Index hot-swapping

RAG와 같은 non-parametric memory  모델의 장점은 테스트 과정에서 지식을 쉽게 업데이트할 수 있다는 점이다. 이를 index hot-swapping이 가능하다고 말한다. 반면 T5나 BART와 같은 parametric-only  모델은 추가적인 학습이 필요하다. 

이를 증명하기 위해 DrQA와 2016년 12월의 위키피디아 덤프를 이용하여 인덱스를 구축하였고 RAG과 출력을 비교하였다.

두 모델의 출력 mismatch는 12%였는데 이는 RAG의 world 지식이 non-parametric memory를 교체하는 것만으로 업데이트 할 수 있음을 의미한다.

**Effect of Retrieving more documents**

![](https://velog.velcdn.com/images/adsky0309/post/31ede7f0-2d22-4c92-8275-edc1b920ea4e/image.png)


모델은 5개 또는 10개의 문서를 이용하도록 순련되었는데 두 경우의 성능차이는 발견되지 않았다. 해당 그림은 _**검색한 문서의 개수에 따른 성능을 나타낸 것**_이다.

RAG-Sequence는 대체적으로 문서의 개수가 늘어남에 따라 성능이 더 좋았는데 RAG-Token의 경우 10에서 최고점을 찍고 내려가는 모습을 보였다. 일반적으로는 더 많은 문서를 사용할 수록 높은 점수를 보임을 알 수 있었다.

## 5 Related Work

### Single-Task Retrieval

이전의 연구들은 단독으로 고려할 때 리트비러가 NLP의 다양한 작업의 성능을 향상시킬 수 있음을 보여줬다. 이러한 태스크들은 ODQA, FVR, Fact completion, Long-form QA, Wikipedia article generation, dialogue, 번역, 모델링 등이 포함된다. 이 논문의 연구는 여러 태스크에서 성공적으로 사용된 방법을 하나의 방식으로 통합해 리트리버에 기반한 단일 아키텍쳐가 여러 태스크에서 최고의 성능을 보임을 입증했다.

### General-Purpose Architectures for NLP

일반적인 목적의 아키텍쳐에 대한 연구는 리트리버 없이고 NLP 태스크에서 좋은 성능을 보였다. 해당 논문은 사전 학습된 생성 언어 모델을 보강하기 위해 리트리버를 사용하여 단일 통합 아키텍쳐로 가능한 작업을 확장하고자 한다.

### Learned Retrieval

정보 검색에서 문서를 검색하는 학습에 대한 중요한 작업이 있으며 최근에는 본 논문과 유사한 사전학습된 모델을 사용한다. 이는 QA나 강화학습과 같은 구체적인 하위 태스크를 목적으로 검색을 최적화하였으며 이 성공은 다양한 검색기반 아키텍쳐와 최적화 기술을 사용하여 단일 작업에서 강력한 성능을 달성하는 반면 단일 검색 기반 아키텍쳐는 미세조정될 수도 있다는 것을 보여준다.

### Memory-based Architectures

문서 인덱스는 메모리 네트워크와 유사하게 신경망이 참여할 수 있는 큰 외부 메모리로 볼 수 있다.  다른 연구세어는 RAG와 같이 원천 텍스트를 검색하는 대신 학습된 임배딩을 검색하는 방법을 사용한다. 또 다른 연구는 fact embedding에 집중하여 사실적인 텍스트를 생성하는 방식으로 대화 시스템의 성능을 개선하기도 하였다. 메모리의 주요 특징은 원시 텍스트로 이뤄져 있기 때문에 사람이 읽고 쓸 수 있어, 모델을 해석하고 업데이트할 수 있다는 점이다.

### Retrieve-and-Edit Approaches

본 논문은 주어진 입력에 대해 유사한 input-output 훈련 쌍을 검색하고 이를 편집하여 최종 출력을 제공한다는 점에서 Retrieve-and-edit style과 유사하다.

이러한 접근법은 기계 번역 및 semantic parsing을 포함한 여러 도메인에서 성공적인 것으로 입증되었기에 이는 RAG 기술이 이러한 세팅에서 잘 작동할 것임을 보여준다.

<br>

## 6 Discussion

이 연구는 parametric과 non-parametric memory를 사용하는 하이브리드 생성 모델을 제안한다. RAG 모델이 ODQA 태스크에서 최고 성능을 보임을 보였고 인적 평가를 통해 RAG의 생성 결과가 BART의 보다 사실적이고 구체적이라는 것을 보였다.  Learned retrieval component에 대한 빈틈없는 조사를 거쳤고 그것의 효용성을 입증하였고 어떻게 리트리버 인덱스가 재학습 없이 ho-swapped를 통해 업데이트 되는지를 보여주었다.

향후에는 사전학습 모델과 리트리버가 처음부터 함께 학습될 수 있는지에 대한 연구를 진행하여 더 좋은 성능의 모델을 구축할 수 있을 것이다.

이 연구를 통해 parametric과 non-parametric 메모리가 상호작용하고 효율적으로 결합하는지에 대한 새로운 연구 방향이 열렸으며 이는 다양한 분야의 NLP 테스크에 적용될 것을 약속한다.

<br>

## 조각
하나의 논문을 이해하기 위해 정말 많은 자료들을 봐야 했다. 다른 사람의 리뷰를 보는 것만으로는 이해가 되지 않았다. 점차 많은 논문을 읽으면서 이렇게까지 자세하게 분석할 일은 없겠지만, 적어도 중요한 논문들은 완전히 이해해야한다고 생각했다. 그럼에도 불구하고 이해하지 못한 것들이 많고(특히 Related Work), 이 논문에만 매달리고 있으면 나아갈 수 없다는 생각에 우선 이 정도로 정리하기로 했다. 좀 더 공부하고 다시 돌아와보자. 보이는게 있을 테니까.


<br>


## 참고

- AIKU :  https://www.youtube.com/watch?v=efoqG3Hg0Ng
- https://gbdai.tistory.com/67
- [https://basicdl.tistory.com/entry/논문리뷰-Retrieval-Augmented-Generation-for-Knowledge-Intensive-NLP-Tasks](https://basicdl.tistory.com/entry/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Retrieval-Augmented-Generation-for-Knowledge-Intensive-NLP-Tasks)