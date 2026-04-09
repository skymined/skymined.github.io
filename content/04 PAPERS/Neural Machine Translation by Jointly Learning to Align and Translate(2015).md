---
tags:
  - paper/DL
created: 2026-03-01T14:32:00
---
![](https://velog.velcdn.com/images/adsky0309/post/1503ef19-603e-447d-9f76-d73355caff93/image.png)


딥다이브 정기세션 첫 번째로 읽게 된 논문이자 Attention Mechanism의 시초라고 볼 수 있다는 논문을 리뷰해보려고 한다. 논문 리뷰는 엄청나야 한다는 생각에 시도도 못하고 있었는데 그냥 자유롭게 쓰면 된다는 딥다 멘토님의 조언에 따라 일단은 무작정 시작해보았다.
논문에 익숙하지 않기 때문에 처음에는 천천히 논문을 따라가면서 이해해보려고 한다.

---

## 개요
신경망 기계 번역(Neural machine translation)은 기계번역 분야에 있어 2015년에는 새롭게 제안된 접근법이었다. 이 논문에서는 기본적인 인코더-인코더 아키택쳐의 성능을 향상시키는데 있어 고정된 길이의 펙터를 사용하는 것이 병목현상을 일으킨다고 추측하며, 모델이 이러한 부분을 명시적으로 하드 세그먼트로 형성할 필요 없이 대상 단어 에측과 관련된 소스 문장의 일부를 자동으로 검색할 수 있도록 하여 이를 확장할 것을 제안한다.

이해한 바로는, 보통의 Encoder-Decoder 모델의 경우에는 인코더가 입력 시퀀스를 고정된 길이의 벡터에 매핑하고, 이 매핑된 하나의 벡터를 바탕으로 디코더나 출력 시퀀스인 번역문을 출력한다. 그런데 이렇게 고정된 길이의 벡터를 사용하면 병목 현상(한 시스템에 부하가 많이 걸려 전체 시스템에 영향을 미치는 것)을 일으킨다. 이 문제를 해결하기 위해 새로운 단어를 생성할 때마다, 입력한 문장에서 관련된 정보를 가지고 와서 Decoder에 모두 넣어주는 것이다.


## Introduction
Neural Machine Translation은 대부분 Encoder-Decoder 형식으로 이루어져 있다. Encoder에서 입력문장을 하나의 고정된 길이 벡터로 변환시키고, Decoder에서는 이 벡터를 이용해 번역 결과를 생성한다.
모든 Encoder-Decoder 시스템에서 정확한 번역의 가능성을 최대화하하는 방식으로 학습된다.
그런데 이렇게 하나의 고정된 길이의 벡터로 생성하면 문제가 발생하고, 이는 특히 문장의 길이가 길어질 수록 심각하게 나타난다.

때문에 해당 논문에서는 Encoder-Decoder 모델에서 새로운 구조를 추가해 성능을 향상할 수 있는 방법을 제안하고자 한다.제안된 모델은 단어를 생성할 때마다 가장 관련성이 높은 정보가 집중되는 소스 문장의 위치 집합을 검색한다. 그리고 모델은 모든 이전에 만들어진 단어들과 위치 정보들과 연관된 context vectors에 기반하여 단어를 예측한다.

이전의 기본적인 Encoder-Decoder와 가장 구별되는 점은 모든 문장이 하나의 고정된 벡터에 인코딩되지 않는다는 점이다.

<br>

## 2 Background : Neural Machine Tranlation

### 2.1 RNN Encoder-Decoder
이 파트는 기본적인 RNN 구조를 보여줬다.
Encoder-Decoder Framework 내에서 Encoder는 입력 문장을 고정된 벡터 C^2로 바꾼다. RNN을 이용하기 위한 가장 일반적인 접근법은 다음과 같다.
여기서 h_t는 time t에서의 hidden state이고, f는 LSTM function, q는 forward RNN을 의미한다.

$$
\begin{aligned}
&h_t=f(x_t,h_{t-1})\\
&c=q({h_1,...,h_{T_x}})
\end{aligned}
$$

디코더는 context vector c와 이전에 예측된 단어들이 주어졌을 때 다음 단어인 y_t'를 예측하도록 훈련된다. 아래와 같은 결합확률분포로 나타낼 수 있다.
여기서 결합확률분포(Joint Probability Distribution)은 두 개 이상의 확률 변수가 동시에 특정한 값 또는 범위에 속할 확률을 나타내는 분포로 두 개 이상의 확률변수들의 교집합이라고 볼 수 있다.
그러니까 특정 이벤트가 동시에 일어나는 경우를 확률로 나타낸 것.

$$
p(y)=\prod^T_{t=1}p(y_t|({y_1,...,y_{t-1}}),c)
$$

RNN을 이용한다면, 각각의 조건부 확률은 아래 식으로 계산될 수 있다. 이때 g는 잠재적으로 다중 layer를 가지고 있는 nonlinear function이다.

$$
p(y_t|(y_1,...,y_{y-1}),c)=g(y_{t-1},s_t,c)
$$

<br>

## 3 Learning to Align and Translate
Attention 기법에 대해서 설명하는 장이었다. (앞으로 읽어야 할 Attention is all you need가 생각났음) 이 논문에서는 attention이라는 용어 대신 align을 사용하는 것 같았다. align은 나란히 만들다, 일직선으로 하다, 라는 뜻을 가지고 있으니 정렬이라고 하면 되겠지만 이렇게 해석하면 말이 조금 이상해지는 것 같아 align을 쓰기로 했다.

3.1 은 decoder가 align기법을 사용한다는 것, 3.2는 incoder가 bidirectional RNN을 사용한다는 것을 설명한다.

### 3.1 Decoder : General Description
새로운 모델에서 조건부 확률은 다음과 같이 정의된다.

$$
p(y_i|y_1,...,y_{i-1},x)=g(y_{i-1},s_i,c_i)
$$

이때 기존의 모델과 다른 점은, context vector가 time에 종속되어 c_i로 표현된다는 점이다. 그러니까 기존의 encoder-decoder 모델에서는 encoder에서 출력된 하나의 고정된 context vector가 decoder의 모든 곳에서 사용되었는데 이 attention mechanism을 사용하면 decoder의 각 time step 마다 서로 다른 context vector가 계산된다. 그러니 모든 단게에서 encoder의 전체 출력에 대한 가중치(alignment weight)를 다르게 부여한다는 소리다.
(시간에 따라 context vector가 변화한다는 점이 가장 큰 차이점)

아래 context vector에는 decoder가 source sentence에서 어떤 위치에 있는 단어에 더 attention을 주어야 하는지에 대한 정보가 담겨있다. (encoder의 모든 hidden state에 대한 weighted sum)
$$
\begin{aligned}
&c_i=\sum^{T_x}_{j=1}\alpha_{ij}h_j\\
&\alpha_{ij}=\frac{exp(e_{ij})}{\sum^{T_x}_{k=1}exp(e_{ik})}\\
&e_{ij}=a(s_{i-1}, h_j)
\end{aligned}
$$
그리고 그에 따른 가중치는 위와 같았다.

### 3.2 Encoder : Bidirectional RNN for Annotation Sequences
일반적인 RNN은 입력 시퀀스를 앞에서부터 순서대로 읽는다.
해당 모델의 encoder는 입력 문장에 대해 이전 내용과 이후 내용을 함께 고려하기 위해 BiRNN을 사용했다.
BiRNN은 forward RNN과 backward RNN을 포함한다. 순방향 RNN과 순방향 은닉 상태, 역방향 RNN과 역방향 은닉 상태를 계산해서 이를 연결하면 다음과 같이 표현할 수 있다.
$$
h_j=[\overrightarrow{h}^T_j;\overleftarrow{h}^T_j]^T
$$

<br>

## 4 Experiment Settings
### 4.1 Dataset
WMT'14의 English-French parallel corpus를 사용하였다.
토크나이징을 한 후 각 언어의 30,000개 가장 빈번한 단어들을 모델 학습에 사용했다.

### 4.2 Models
두 가지 타입의 모델을 학습하였다.
하나는 RNN Encoder-Decoder모델이고 다른 하나는 RNNsearch로 각 모델을 최대 30개의 단어로 구성된 문장, 최대 50개의 단어로 구성된 모델로 분류하였고 이를 -30, -50로 표현했다.
RNNencdec-30, RNNencdec-50, RNNsearch-30, RNNsearch-50
RNNdencdec는 1000개의 hidden state를 가지고 있으며 RNNsearch는 forward rnn과 backward rnn이 각각 1000개의 hidden state를 가지고 있다. 디코더 역시 1000개의 hidden state를 가지고 있다.
해당 모델은 5일 동안 학습을 진행했다.

<br>

## 5 Results
### 5.1 Quantitative Results
![](https://velog.velcdn.com/images/adsky0309/post/7a616d9b-0e83-4db2-a697-672316ff8ea1/image.png)

BLEU score을 바탕으로 성능을 평가하였다. RNNsearch가 RNNencdec보다 훨씬 뛰어난 성능을 보였으며 심지어 RNNsearch-30이 RNNencdec-50보다 성능이 뛰어났다.

![](https://velog.velcdn.com/images/adsky0309/post/515b3692-aedf-43c5-a5ee-221154539dd8/image.png)

Figure2를 보면, 문장의 길이가 길어질 수록 성능이 떨어진다는 것을 볼 수 있다. 그런데 RNNSearch-50의 경우에는 길이가 길어져도 성능 저하가 없다.

### 5.2 Quantitative Analysis
아래는 입력과 출력 문장의 각 단어에 대한 Alignment를 시각화한 표이다. 그러니까 번역 과정에서 source sentence의 어떤 단어를 참조했는지 볼 수 있다.
검은색이 0, 하얀색이 1이라 두었을 때, 1에 가까울 수록 alignment 점수가 높은 것이다.
그림에서 볼 수 있듯 문장 구조가 비슷하기 때문에 대부분 직선형태이지만 그 안에서 세부적인 차이가 있는 경우에는 단어의 순서보다 꼭 필요한 단어를 참조하는 것을 알 수 있다.

![](https://velog.velcdn.com/images/adsky0309/post/df5511dc-5981-477f-8dd6-55985e7430e7/image.png)


## 7 Conclusion & View

결론적으로 본 논문은 기존 인코더-디코너 NMT 모델의 한계(병목현상)를 극복하기 위해 가변적인 길이의 context vector를 번역을 위해 매번 만들어내는 방식을 사용했다.

![seq2seq 방식 (나동빈 github)](https://velog.velcdn.com/images/adsky0309/post/5b6edb48-3f4b-45f4-ba6c-c42d7e538d0c/image.png) 출처: seq2seq 방식 (나동빈 github)

<br>

논문을 이해하기 위해 기존 seq2seq 방식을 찾아보았는데, 이런 식으로 고정된 크기의 context vector를 사용하면 확실히 정보를 제대로 번역하기 힘들 것 같았다.
어쨌든 해당 논문에서는 이런 제안된 아키텍처가 더 나은 기계 번역과 일반적인 자연어를 이해하기에 유의미한 단계라고 이야기했다.
논문 자체를 이해하긴 했으나 어떤 식으로 작용하는 것인지에 대한 이해가 완전히 이뤄졌는지는 잘 모르겠다... 다른 논문이나 개념을 쉽게 풀어쓴 책을 읽은 후에 다시 한 번 논문을 봐야겠다고 다짐하며 첫 논문 리뷰(라고 말하고 해석이라 읽는) 끝.
(추가로 깨닫는 사실이 있으면 업데이트할 예정)


## 8 멘토님의 질문에 답해보자

> ❓ 왜 특정 단어의 의미를 표현할 때 주변 N개의 단어(context window)를 주로 사용할까요?

단어의 의미는 주변의 단어 조합에 따라 달라질 수 있기 때문이다. 단어의 진짜 의미를 알기 위해서는 전후 문맥을 파악해야만 한다.
예를 들어 '아이가 불을 켰다'와 '누군가가 숲에 불을 질렀어'라는 문장에서 '불'이 가지고 있는 의미는 다르다.


> ❓ Attention 메커니즘은 Long Term Dependency 문제를 극복했다는 것 이외에 어떤 의의를 가질까요?
긴 시퀀스를 효과적으로 처리할 뿐만 아니라 특정 정보에 집중할 수 있고 어텐션 가중치를 시각화할 경우 모델이 어느 부분에 집중했는지도 알 수 있다는 점에서 발전 가능성이 있다고 생각한다.

<Br>

---

<Br>
  

#### 참고자료1
https://basicdl.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Neural-Machine-Translation-by-Jointly-Learning-to-Align-and-Translate

#### 참고자료2
https://blog.naver.com/mewmew16/223159664787

