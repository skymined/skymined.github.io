---
tags:
  - paper/DL
created: 2026-03-01T14:32:00
---
![](https://velog.velcdn.com/images/adsky0309/post/983c2e4e-7a2f-4285-a053-d924de351cad/image.png)

본 논문은 자연어 이해 태스크에서 강력한 성능을 달성하기 위해 생성 사전 학습과 판별 미세조정을 결합하는 프레임워크를 소개한다.

## 1. Introduction
지금까지의 NLP task들은 manually labeled data를 필요로 했다. unlabeled data 로 모델을 학습시켜 시간 소모나 가격 측면에서 효과적이었고 사람들이 알지 못하는 데이터의 특성까지 모델이 학습하게 하고, 작은 수정만으로 효과적인 transfer를 가능하게 하였다.

그러나 unlabeled data로 학습하는 데에는 어려움이 있었는데 먼저 1) 최적화해야할 목적함수(optimization objective)가 무엇인지 명확하지 않았다. 예로 지도학습에서는 레이블된 데이터를 사용하므로 오차를 최소화하는 것이 목적함수이다. 2) 또한 비지도 학습으로 학습된 representation을 실제 목표 과제(downstream task)에 효과적으로 전달하는 방법에 대한 합의된 의견이 없다.

이에 본 논문에서는 unsupervised pre-training과 supervised fine-tuning을 결합한 semi-supervised 사용하였다. unlabeled data를 LM에 사용해서 초기 파라미터를 설정하고 supervised objective에 해당하는 target task에 적용하였다. 이때 transformer 구조를 사용하여 RNN의 고질적인 문제점인 Long-term dependency 문제를 해결하였다.


## 2. Related Work
### Semi-supervised learning for NLP
몇 년간 연구자들은 word embedding의 장점을 입증하였다.
>🍀 Word Embedding이란?
유사한 의미를 가진 단어들을 밀집 벡터 공간상에 위치시켜 단어를 연속적인 실수 벡터로 표현하는 기법으로 자연어 처리 분야에서 매우 유용하게 사용된다.
⇒ 단어의 의미를 보다 잘 알 수 있고 서로 간의 유사도를 알 수 있으나 단어의 다의성, 문맥 무시 등의 한계로 최근에는 contextualized word representation 등이 발전하고 있다.


### Unsupervised pre-training(비지도 사전 훈련)
Unsupervised pre-training은 supervised training의 학습 목표를 수정하는 것이 아니라 지도 학습을 위한 좋은 초기화 지점, 그러니까 모델의 가중치 초기값을 찾는 것이 목표이다. 후속 연구에서는 pre-training이 정칙화 방식(regularization)으로 작용하여 심층 신경망에서 더 나은 정규화가 가능하다는 것을 보여주었다. (정칙화 : 모델의 형태를 최대한 간결하게 만들어 과적합을 방지함)

최근 연구에서는 이 방법이 이미지 분류, 음성 인식, 엔티티 명혹화 등의 다양한 업무에서 심층 신경망을 훈련하는데 도움이 되었다.

이전 연구의 경우에는 LSTM을 사용하였기 때문에 짧은 범위의 예측만이 가능했으나 본 논문의 경우에는 긴 길이의 언어 구조를 포착할 수 있는 Transformer를 사용하였다는 점이 차이이다.

( LSTM과 같은 순환 신경망 기반 언어 모델은 문맥 창의 크기가 제한적이라 먼 거리의 단어 간 관계를 포착하기 어려웠다는 한계점이 있다.)
반대로 transformer는 더 긴 길이의 언어 구조를 포착할 수가 있다.

### Auxiliary training objectives
레이블이 없는 unlabeled 데이터에 보조적으로 unsupervised training objective를 추가해주는 것으로 semi-supervised learning(준지도 학습)의 대안이 될 것이다.
즉, labeled data에는 지도 학습 목표로 unlabeled data에 대해서는 비지도 보조 목표를 함께 학습해 준지도 학습과 유사한 효과를 기대하는 것이다.
본 연구는 보조 목표를 사용하긴 하지만, unsupervised pre-training은 이미 목표 작업과 관련된 언어적 측면을 학습한다.

<br>

## 3. Framwork
학습 과정은 두 가지 절차를 밟는다. 하나는 대규모 택스트 모음에서 대용량 언어 모델을 학습하는 것이다. 두 번째는 라벨링된 데이터를 가지고 식별 작업에 모델을 적용하는 파인튜닝 단계이다.

### 3.1 Unsupervised pre-training
아래 수식은 standard language modeling object이며 이때 k는 context window의 크기이다. parameter theta는 SGD로 학습한다.

$$
L_1(u)=\sum_ilogP(u_i|u_{i-k}, ..., u_{i-1};\theta)
$$

> 📌 SGD(Stochastic Gradient Descent)란?
모델의 가중치를 업데이트하는 데 사용되는 optimization 알고리즘. 기본적으로 경사 하강법의 한 변종이며 전체 데이터셋이 아닌 일부 미니배치만을 사용하여 가중치 업데이트를 수행하는 확률적 방법


본 연구의 실험에서는 언어 모델을 위해 다층의 transformer decoder를 사용하였다.
Transformer는 입력 시퀀스에 대해 multi-headed self-attention 연산을 적용하여 토큰 간의 관계를 모델링하고 이를 전향신경망 레이어(position-wise feedforward layer. FFN 레이어)에 통과시켜 최종 출력 분포를 내놓는 구조이다.

$$
\begin{aligned}
&h_0=UW_e+W_p \\
&h_l=transformer\_block(h_{l-1})\forall i\in[1,n]\\
&P(u)=softmax(h_nW_e^T)
\end{aligned}
$$

U는 토큰의 문맥 벡터, n은 층의 개수, W_e는 토큰 임베딩 행렬, 그리고 W_p는 위치 임배딩 행렬이다.

>📌 **Position-wise feedforward layer(전향 신경망 레이어)란? 잘 모르겠음…. 추가적으로 알아봐야할 것 같다**

<br>

### 3.2 Supervised fine-tuning
첫 번째 단계를 지나고 나면 우리는 지도 학습된 목표에 파라미터를 적용시켜야 한다.
사진 설명을 입력하세요.

$$
\begin{aligned}
&P(y|x^1,...,x^m)=softmax(h_l^mW_y)\\
&L_2(C)=\sum_{(x,y)}logP(y|x^1,...,x^m)
\end{aligned}
$$
여기서 $$h_l^m$$ 는 input x가 모델의 마지막 transformer block에서 가지는 activation 값으로 여기서 l은 시퀀스의 길이를 나타낸다.
이 값을 추가적인 linear output layer를 $$W_y$$와 함께 통과 할 때의 parameter곲에 softmax를 취한 값이 확률이다.
이때 $$W_y$$는 출력을 목표 클래스 수만큼 변환하는 가중치의 행렬이다.

$$
L_3(C)=L_2(C)+\lambda*L_1(C)
$$

보조 목표로 언어 모델링을 포함시키는 것은 지도 모델의 일반화를 개선하고 수렴을 가속화한다. 때문에 최종적으로 사전학습 모델 기반인 L1과 finetuning 모델 기반의 L2를 가중합한 L3을 사용하기로 한다.

<br>

### 3.3 Task-specific input transformations
![](https://velog.velcdn.com/images/adsky0309/post/06c01a63-7818-40bc-ad03-1680b2583a23/image.png)

사전 학습된 모델은 연속 텍스트 시퀀스로 훈련되었기 때문에 구조화된 입력(문장 쌍 등)을 처리하기 위해서는 약간의 수정이 필요하다. 이전 연구에서는 전이된 표현 위에 작업별 아키텍쳐를 학습하는 방식을 제안했으나 이러한 접근법은 상당한 양의 커스터마이징이 필요했고 추가된 아키텍쳐에 대해서는 전이 학습이 적용되지 않았다.
때문에 본 논문에서는 traversal-style 접근법을 사용하여 구조화된 입력값을 정렬된 토큰 시퀀스로 변환하였다. 이를 통해 아키텍쳐를 광범위하게 변경하지 않고 미세조정을 할 수 있다.
>🍀 Traversal-style 접근법
구조화된 입력을 연속적인 토큰 시퀀스로 변환하는 방식으로 사전학습된 언어모델이 처리할 수 있는 형태가 된다.


해당 방법을 사용하면 태스크마다 다른 모델을 사용할 필요 없이 다양한 태스크를 일반적으로 사용할 수 있는 모델을 만들 수 있으며 이 아이디어가 GPT의 핵심이라고 할 수 있다.


## 4. Experitments

### 4.1 Setup
언어 모델을 학습시키기 위해 BooksCorpus dataset을 사용하였다. 이 데이터셋은 모험, 판타지, 로맨스를 포함하여 다양한 장르의 7000개 책을 포함하고 있다. 긴 범위의 연속된 텍스트가 포함되어 있기 떄문에 long-range 정보를 활용할 수 있다.
대조적으로 ELMo에서 사용되었던 1B Word Benchmark가 있는데 이는 문장의 수준이 섞여 있어 long-range structure을 파괴하였다.

### 4.2 Supervised fine-tuning
![](https://velog.velcdn.com/images/adsky0309/post/8fd32202-d0f3-4ea5-812d-0a17c71cb173/image.png)

자연어 추론, QA, 의미 유사도, 텍스트 분류를 포함한 다양한 지도학습 과제들을 학습했다. Table1 은 모든 과제와 데이터셋의 개요를 보여준다.


#### Natural Language Inference
![](https://velog.velcdn.com/images/adsky0309/post/3b82d884-1432-465a-8853-5ce5f3bb6205/image.png)

짝지어진 두 개의 문장의 관계를 파악하는 문제로 entailment, contradiciton, neutral 중 하나로 선택된다.

>📌 Entailment?
자연어 처리에서 entailment는 두 문장 또는 명제 간의 논리적 관계를 나타내는 개념이다.
1.  Entailment(참말) 전제 문장을 가정했을 때 가설 문장이 반드시 참이 되는 경우
2. Contradiction(모순) 전제 문장을 가정했을 때 가설 문장이 반드시 거짓이 되는 경우
3. Neutral(중립) 전체로부터 가설의 참/거짓 여부를 판단할 수 없는 경우


MNLI, SNLI, SciTail, QNLI, RTE 라는 5개의 dataset을 평가하였고 이에 RTE 를 제외하고는 유의미한 성능향상을 보였음을 알 수 있었다.

#### QA(Question answering and Commonsense reasoning)
![](https://velog.velcdn.com/images/adsky0309/post/09c1e539-afd2-4f92-be7a-2ff357342a89/image.png)

Story Cloze Test는 두 가지 선택지 중 맞는 끝문장을 고르는 문제이고 RACE dataset은 실제 시험 문제로 구성되어 있으며 지문의 길이가 매우 다양해 모델의 이해력을 폭넓게 평가할 수 있다. 표에 의하면 Finetuned Transformer LM이 모든 test에서 좋은 성능을 발휘함으로 긴 범위의 문맥을 효과적으로 잘 다룬다는 것을 증명한다.

#### Semantic Similarity & Classification
![](https://velog.velcdn.com/images/adsky0309/post/c0b6b9c4-985b-4c9d-bde7-346a824b7402/image.png)

Semantic Similarity task는 두 문장이 의미적으로 같은지 아닌지를 예측한다.
MRPC, STSB, QQP라는 세 가지 dataset을 이용하였으며 QQP에서 유의미한 개선을 보였다.
Classification의 경우 CoLA에서 큰 개선을 보였다.
또한 추가적으로 GLUE score가 68.9dptj 72.8로 개선되었음을 볼 수 있다.
전체적으로, 본 모델은 12개의 dataset 중 9개의 dataset에서 최고 성능 결과를 달성하였다. 또한 작은 규모의 STS-B 데이터셋부터 가장 큰 SNLI 데이터셋까지 다양 한 규모에서 좋은 성능을 내는 것으로 보아 데이터셋의 규모와 상관없이 잘 작동하였다.

<br>

## 5. Analysis
#### Impact of number of layers transferred
![](https://velog.velcdn.com/images/adsky0309/post/ae8d4218-8313-437e-b077-d95f21340c3d/image.png)

비지도 학습에서 지도 학습으로 전이하는 층의 개수에 따라 성능이 향상된다는 것을 알 수 있다. 요컨데 사전 학습 모델에서 점점 더 많은 층을 전이할 수록 성능이 향상되었다.
사전 학습된 언어 모델이 있을 때 이 모델의 일부 층을 가지고 와서 새로운 작업을 위한 모델을 추가하는 것이 층을 전이한다는 것의 의미이다.
즉, 사전 학습된 모델의 일부을 재사용하여 새로운 작업에 대한 성능을 향상시킨다는 것인데, 그 층이 많을 수록 성능이 향상된단 뜻이다.

#### Zero-shot Behavior
논문의 저자는 지도 학습 파인튜닝 없이 생성 모델만을 사용하여 작업을 수행하는 휴리스택 솔루션을 설계했다. FIgure2의 우측을 보면 pre-training된 생성모델의 성능을 볼 수 있다. 휴리스틱의 성능이 안정적이며 꾸준히 향상된다는 것을 볼 수 있으며 이때 LSTM과 비교하였을 때 분산이 작다는 점을 들어 Transformer 아키텍쳐가 LSTM에 비해 전이 학습에 적합하고 다양한 작업에서 일관된 성능을 보임을 주장할 수 있다.
>📌 Zero-shot Behavior
기계 학습 모델이 특정 작업에 대한 명시적인 학습 없이도 해당 작업을 수행하거나 처리할 수 있는 능력으로 모델의 일반화 능력을 나타냄

#### Ablation studies
![](https://velog.velcdn.com/images/adsky0309/post/14d8b3db-f863-4a90-af4e-729a29807e8e/image.png)

Ablation은 모델의 구성 요소를 순차적으로 제거하거나 비활성하여 해당 구성 요소가 전체 모델의 성능에 미치는 영향을 평가하는 방법이다.
온전한 모델의 구성요소는 Transformer, pre-training, auxiliary LM이며 각각을 제외하며 해당 구성요소의 영향을 알아봤다.
첫번째로 pre-training을 하지 않은 모델의 경우 전체 score가 14/8%나 감소하는 경향을 보였고 Transformer대신 LSTM을 사용한 모델의 경우 5.6% 감소했다.
Auxiliary Language Modeling을 제외한 경우에는 전체 스코어는 0.3 증가하였으나 NLI과 QQP에서는 감소하는 모습을 볼 수 있었다.


## 6. Conclusion
본 논문은 사전 학습과 이와는 차별적인 파인튜닝을 통해 단일 task에 국한되지 않는 자연어 처리 이해가 가능한 프레임워크를 소개했다.
12개 중 9개의 데이터셋에서 성능향상을 보인 이 모델은 QA, Semantic similarity assessment, entailment determinaiton, text classification 등의 판별적 작업에 상당한 세계 지식과 장거리 종속성 처리 능력을 성공적으로 전이했다.
이러한 연구 결과는 상당한 성능 향상이 가능하며 이러한 접근 방식에 가장 맞는 모델과 데이터셋에 대한 힌트를 제공한다.
연구진들은 이 모델이 자연어 처리와 다른 도메인에 있어 비지도 학습의 새로운 연구를 가능하게 하고 나아가 비지도 학습이 언제 그리고 어떻게 작동하는지에 대한 이해를 개선하게 할 것이라 말한다.


## 8. TAI
### 논문에 대한 견해
해당 모델의 특징으로는 레이블이 없는 사전 학습, 특정 태스크에 대한 지도 미세조정(파인튜닝), Transformer achitecture 그리고 쿠고적 입력을 연속 시퀀스로 변환하는 것으로 이해했다. 또한 12개 중 9개의 dataset에서 최고 성능을 달성하였다.
이는 단일 사전 학습 모델만으로도 다양한 태스크에서 우수한 성능을 보일 수 있음을 입증하였다. 완전히 단일 모델로만 끝까지 간 것은 아니지만(파인튜닝을 거침) 사전학습 과정을 통해 다부분의 파라미터를 효과적으로 초기화하고 약간의 미세조정만으로도 다양한 태스크에서 높은 성능을 보였다는 것에 의의를 가진다.
즉, 이전에는 각 태스크 별로 모델을 완전히 새로 설계했던 것에 비해 효율적이라고 볼 수 있다.



### Question

>❓ 통계적 언어 모델과 GPT 모델은 모두 같은 training objective를 갖습니다. (sub sequence가 주어졌을 때 조건부 확률에 의해 다음 토큰을 예측) 그러면 신경망 기반 언어 모델은 통계적 언어 모델과 어떤 차이를 가질까요?

통계적 언어 모델은 통계적 기법을 사용해서 언어를 모델링하는 방식으로 대표적으로 n-gram 방식이 있다. n-gram 모델은 이전 n-1개의 단어 시퀀스가 주어졌을 때, 그 다음에 올 단어의 조건부 확률을 모델링한다. 반면 신경망 기반 언어 모델은 RNN이나 Tranformer 등의 딥러닝 모델을 통해서 언어를 모델링한다. 입력된 단어 시퀀스를 Embedding 벡터로 변환하고 신경망에 통과시켜 다음 단어의 예측 확률 분포를 출력한다.

>❓ GPT-1의 pre-training, fine-tuning 단계에서의 학습 목표는 어떻게 다른가요?

Pre-training(사전 학습) 단계에서는 라벨링되지 않은 대량의 언어 모델을 학습하며 조건부 확률 P(토큰|이전 CONTEXT)를 최대화하는 것이 목표이다. (= 다음에 올 단어 토큰이 무엇인지 가장 잘 예측하도록 모델을 학습시킨다는 뜻) 여기서 학습된 지식은 downstream 과제에서 전이되어 높은 성능을 내게 해준다.

Fine-tuning(미세 조정) 단계에서는 각 태스크 별 지도학습 데이터셋으로 모델을 미세조정한다. 때문에 해당 태스크의 손실함수를 최소화하는 것이 목표이다.

>❓ GPT와 BERT의 학습 특성을 고려하여 장단점을 비교해주세요.

GPT와 BERT는 둘 다 Transformer 아키텍쳐를 기반으로 한다.
그러나 BERT는 양방향 정보를 활용해 문맥을 파악하는데 중점을 둔 반면, GPT는 주어진 정보를 바탕으로 다음에 올 내용을 예측하는 것에 중점을 두었다.
BERT는 자연어 이해(NLU)에 특화되었다면 GPT는 자연어 생성(NLG)에 특화되었다고 할 수 있다.
때문에 BERT는 주변의 문맥 정보를 앞뒤 양방향에서 동시에 파악하므로, GPT에 비해 단어나 구문의 정확한 의미를 더 잘 이해할 수 있다. 다만 양방향 특성상 계산 복잡도가 높고 고정된 길이의 입력만 처리하기 때문에 장문 처리에 어려움이 있다. 반면 GPT는 이러한 context 길이에 제한이 없기 때문에 장문의 의미를 잡아내는데 유리하다.

---

#### 참고자료1
https://brunch.co.kr/@harryban0917/280
