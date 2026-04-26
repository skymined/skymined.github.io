---
tags:
  - paper
created: 2026-03-01T14:32:00
---

## SUMMARY

해당 논문은 EEG(Elctroencephalogram) 기반 MI(Motor Imagine, 운동상상) 분류를 위한 어탠션 기반 시간 합성곱 네트워크, **ATCNet 모델을 제안**한다.

ATCNet 모델은 **원시 MI-EEG 신호를 압축된 시간 순서로 인코딩하는 CV 블록**, **시간 시퀀스에서 가장 효과적인 정보를 강조하는 멀티헤드 셀프 어탠션 블록**, 그리고 **고수준 시간적 특징을 추출하는 TC블록**으로 이루어져 있으며 각 블록들은 모델에 높은 기여를 하는 것으로 보인다.

제안된 모델은 BCI Competition IV-2a 데이터셋에서 최신 기술들을 능가하는 성능을 보였으며 적은 파라미터를 통한 결과라는 점에서 **제한된 자원을 가지고 있는 산업 디바이스에서의 활용도가 높을 것**이라는 점에서 의의를 가지고 있다.

하지만 **BCI Competition IV-2a 데이터셋 하나만을 가지고 평가**했다는 점에서 해당 모델에 확신을 가지기 위해서는 또 다른 데이터셋을 통한 평가자료의 필요성을 느낀다.

---

# Paper Review

## Abstract & Word

해당 논문은 EEG(Electroencephalogram)기반 운동상상 분류를 위한 어탠션 기반 시간 합성곱 네트워크(ATCNet)를 제안한다. 이떄 EEG란 뇌파도, 즉 뇌 내의 전위 변화를 기록한 파형이다.


>📎 뇌파?
우리 뇌의 신경세포는 긴 줄로 연결되어 있고, 전기의 흐름에 의해 자극이 전달되고 신경 말단부에서는 신경 전달물질에 의해 다른 세포로 연결이 된다. 이런 과정에서 뇌 표면에 발생되는 특수한 전기 파동이 뇌파!



ATCNet은 다음과 같은 기술들을 활용한다.

- 다중헤드 자기주의를 사용하여 MI-EEG 데이터의 중요 특징 강조
- 시간 합성곱 네트워크로 고수준 시간적 특징 추출
- 합성곱 기반 슬라이딩 윈도우로 MI-EEG 데이터 효율성 증강


>📎 다중헤드 자기주의(Multi-head Self-attention)
자기주의 매커니즘의 확장된 형태로 입력 데이터를 여러 개의 ‘헤드’로 나누어 각각 독립적으로 주의 계산을 수행하는 것을 말한다. 각 헤드는 입력의 다른 측면에 집중할 수 있어 더 다양한 특징을 포착할 수 있으며 최종적으로 각 헤드의 결과를 결합하여 더 풍부한 표현을 얻을 수 있다.
이때, 자기주의 매커니즘은 긴 거리 의존성을 효과적으로 포착할 수 있고 병렬 처리가 가능하여 계산 효율성이 높다. 때문에 시간적으로 멀리 떨어진 EEG 신호 간의 관계를 포착할 수 있다.


>📎 MI-EEG 데이터(Motor Imagery EEG data)
MI는 운동 상상, EEG는 뇌전도를 의미한다. 즉, MI-EEG 데이터는 사람이 특정 신체 부위의 움직임을 상상할 때 뇌에서 발생하는 전기적 신호를 기록한 데이터다.
이 데이터의 중요 특징으로는 특정 주파수 대역의 변화, 특정 뇌 영역의 활성화 패턴, 시간에 따른 신호 변화 패턴, 채널 간 상관관계 등이 있다.


>📎 시간 합성곱 네트워크(Temporal Convolutional Network, TCN)
시계열 데이터 처리를 위해 설계된 특별한 형태의 합성곱 신경망이다. 일반적인 CNN과는 달리 인과적 합성곱(casual convolution)을 사용하여 미래 정보가 과거에 영향을 주지 않는다.


>📎 합성곱 기반 슬라이딩 윈도우(Convolutional-based Sliding Window)
전통적인 슬라이딩 윈도우 기법을 합성곱 연산으로 구현한 방법이다.
슬라이딩 윈도우 기법(Sliding WIndow Technique)은 긴 시퀀스 데이터를 처리할 때 사용되는 방법으로, 고정된 크기의 ‘윈도우’를 데이터 위에서 이동시키면서 분석하는 기법이다. 이때 윈도우는 한 번에 처리할 데이터의 양을 결정하고 각 윈도우의 위치에서 동일한 연산을 수행하게 된다. 긴 시퀀스를 작은 조각으로 나누어 처리할 수 있기 때문에 연속적인 EEG 신호를 일정 길이의 세그먼트로 나누어 분석할 수 있다는 장점이 있다. 또한 시간에 따른 EEG 신호의 변화를 포착할 수 있다.

<br>


## 1. Introduction

뇌-컴퓨터 인터페이스(BCI)는 뇌 활동을 해석하여 외부 장치를 제어하는 획기적인 기술이며 MI-EEG(운동상상 뇌전도)는 많은 BCI 응용 분야에서 사용된다. 그러나 뇌 신호 해독의 제한적인 성능이 BCI 산업의 광범위한 성장을 제한하고 있다.

해당 논문에서는 EEG 기반 운동상상 분류를 위한 주의 기반 시간 합성곱 네트워크, ATCNet를 제안한다. ATCNet 모델은 과학적 기계학습, 다중헤드 자기주의, 시간 합성곱 네트워크, 합성곱 기반 슬라이딩 윈도우 등 여러 기술을 사용하여 MI 분류 성능을 향상시킨다.

+MI-EEG 신호는 의학적인 것 뿐만 아니라 차량이나 드론 컨트롤, 가상 현실 등에서도 사용될 수 있다. 

## 2. Proposed ATCNet model

ATCNet은 3개의 주요 블록으로 구성된다.

![](https://velog.velcdn.com/images/adsky0309/post/f1597492-11ed-4b45-93f7-669eca8ffc06/image.png)


1. 합성곱(CV) 블록: MI-EEG 신호를 시간 시퀀스로 인코딩
2. 어탠션(AT) 블록: 다중헤드 자기주의를 사용하여 중요한 특징 강조
3. 시간 합성곱(TC) 블록: 고수준 시간적 특징 추출

### A. Preprocessing and Input Representation

해당 논문에서는 전처리하지 않은 로우 MI-EEG 신호를 집어넣었다. 

### B. 합성곱 블록(Convolutional Block)

![](https://velog.velcdn.com/images/adsky0309/post/2363a049-4d7f-4c52-97fa-73feb87563dc/image.png)


시간 축을 따라 필터를 적용하여 특징을 추출하는  CV 블록은 크게 3단계로 나눌 수 있다.

첫 번째 레어어는 먼저 시간 합성곱(Temporal Convolution)으로 이를 통해 필터가 4Hz 이상의 주파수와 관련된 시간 정보를 추출할 수 있다. 그 이후에는 배치 정규화(BN)를 통해 네트워크 학습 속도를 개선한다.

두 번째 레이어는 채널 깊이-방향 합성곱(Channel depthwise convolution)으로 여기서 C는 뇌파 채널의 수다. 각 필터는 단일 시간 특징 맵에서 뇌파 채널과 관련된 공간적 특징을 추출한다. 

세 번째는 공간 합성곱(Spatial Convoluton)으로 여기서 공간적 패턴을 학습한다.

두 번째와 세 번째 레이어에는 배치 정규화와 지수선형단위 활성화가 진행되고 그 이후 평균 풀링 레이어(Avg. Pool)가 뒤따르는데 이를 통해 차원을 축소한다.

### C. Convolutional-Based Sliding Window(SW)

슬라이딩 윈도우는 시간 축에 따라 데이터를 분할하여 처리하기 위해 사용된다. 이를 통해 데이터를 늘리고 디코딩 정확도를 강화할 수 있다. 해당 논문에서는 슬라이딩 윈도우를 합성곱 블록과 통합하였다. 이를 통해 합성곱 계산은 모든 윈도우에서 실행되고, 이는 병렬 처리를 통해 학습과 추론 시간을 줄인다.

### D. Multihead-attention block

딥러닝에서 어탠션 매커니즘은 몇 가지 중요한 요소에만 집중하는 인간의 뇌를 모방하기 위한 노력이다.  Multihead-attention block에서는 중요한 정보는 강요하고 불필요한 정보는 억제함으로서 더 중요한 특징에 집중하기 위한 단계이다.

![](https://velog.velcdn.com/images/adsky0309/post/9b1443db-be17-40e0-b6a7-bcfb3df790ac/image.png)


각각의 self-attention 레이어는 Q, K, V라는 세 가지 메인 구성요소를 가지고 있다.

이때 각 헤드는 Query와 Key의 벡터의 접곱을 계산하고 이 값을 Scaling하는데, scale된 값에 softmax 함수를 적용하여 값에 대한 선택 편향을 알려주는 어탠션 스코어를 생성한다.

이 어탠션 스코어와 value 벡터의 곱셈을 통해 가중합을 구하고 이를 병합한 후 선형 변환을 통해 최종 출력을 하게 된다.

### E. TCN Block(Temporal Convolutional Network)

TCN은 시간적 데이터를 처리하기 위해 설계된 합성곱 신경망으로 두 개의 잔차 블록(Residual Block)으로 이루어져 있고 각 잔차 블록은 두 개의 Dilated Casual Convolutional layer와 그 뒤를 이은 Batch Norm + ELU 활성화로 이루어져 있다.

이때 두 개의 잔차블록을 구성한 이유는 고차원적인 특징을 추출하기 위함이며 모델이 깊어질 수록 발생할 수 있는 기울기 소실 문제 완화에 도움이 된다.

![](https://velog.velcdn.com/images/adsky0309/post/25deb52a-ca91-4a2b-b63f-778cd95cc09c/image.png)


인과적 합성곱은 미래의 정보가 과거의 정보에 관여하는 것을 방지하는데 사용된다. 즉 시간 t에서의 출력은 시간 t와 그 이전의 입력에만 의존하는데, 확장된 인과적 합성곱은 네트워크의 깊이를 증가시키면서 수용 필드를 기하급수적으로 확장할 수 있게 해 긴 시퀀스의 관계를 학습할 수 있다.

![](https://velog.velcdn.com/images/adsky0309/post/e37a44e0-f6f9-40d6-af61-fcbe677ab70d/image.png)


16개의 시간적 요소가 TCN 안으로 들어가고 TCN의 input과 output의 벡터 사이즈는 모두 동일하게 32이다.

![](https://velog.velcdn.com/images/adsky0309/post/ff20df64-fbb9-4c3b-bef4-ee888b0edb35/image.png)


또한 모든 실험에 사용된 하이퍼 파라미터는 다음과 같으며 여러 실험을 기반으로 경험적으로 설정되었다.

<br>

## 3. Experimental results and discussion:

### A. Selected Dataset and Evaluaion Approaches

BCI Competition IV-2a 데이터셋을 사용하여 평가하였다.

BCI-2a는 잘 알려진 MI-EEG 데이터셋으로 2008년에 Graz University of Technology에서 만들어졌다.

### B. Performance Metrics

제안된 모델은 정확도와 Kappa score(K-score)를 통해 평가된다.

$$
ACC=\frac{\sum^n_{i=1}TP_i/I_i}{n}
$$

이때 $TP_i$는 클래스 $i$에서 정확히 예측된 샘플의 수이고 $I_i$는 클래스 $i$에서 샘플의 수이다. $n$은 클래스의 수를 의미한다.

$$
k_score=\frac{1}{n}\sum^n_{a=1}\frac{P_a-P_e}{1-P_e}
$$

$P_a$는 실제 일치도(Actual agreement)를 의미하며 $P_e$는 우연에 의한 예상 일치도(Expected agreement by chance)를 의미한다.

k-socre는 -1부터 1사이의 값을 가지며 0은 우연에 의한 일치와 동일함을 의미한다.

### C. Training Procedure

모델은 Nvidia GTX 2070 8 GB 단일 GPU를 이용하여 TensorFlow Flamework를 통해 학습/테스트되었다.

전체적으로, ATCNet 모델은 85.38%의 정확도와 0.81의 k-score를 가지며 이는 최신 모델과 비교하였을 때 더 높은 성능임을 의미한다.

### D. Contributions of ATCNet Blocks

![](https://velog.velcdn.com/images/adsky0309/post/ff625eeb-b210-4836-be43-0075159f2bd3/image.png)


위의 표는 BCI-2a 데이터셋을 사용하여 MI 분류 성능에 ATCNet 모델의 각 블록이 기여하는 정도를 보여주며 왼쪽은 제거된 블록의 조합을 의미한다.

각 블록을 제거했을 때, 제거하지 않았을 때(None)보다 성능저하가 일어났으며 특히 TC 블록을 제거했을 때 성능 저하가 가장 크게 일어났다.

이는 **모든 블록들이 모델의 성능에 중요한 기여를 한다는 것을 보여준다**.

### E. Varying the Temporal Sequence Length

![](https://velog.velcdn.com/images/adsky0309/post/f671cf67-0ded-449a-97b6-d2ccca26d19f/image.png)


해당 그래프는 17, 18, 28 세 가지 시간 시퀀스를 이용할 때 윈도우, 창의 수가 성능에 미치는 영향을 보여준다. 이때 x축은 1부터 16까지의 창의 수이고 y축은 정확도를 나타낸다.

이때 Tc=20과 Tc=17 시퀀스는 윈도우의 수가 5개일 때 가장 좋은 성능을 보였으며 그 이후부터는 조금씩 감소하는 경향을 보였다.

이를 통해 시퀀스의 길이와 창의 수가 모델의 성능에 중요한 영향을 미친다는 것을 알 수 있다.

### F. Comparing Different Attention Schemes

ATCNet 모델에서 어탠션 매커니즘이 MI 분류 성능에 어떤 영향을 미치는지 보여준다.

![](https://velog.velcdn.com/images/adsky0309/post/3eca714f-dd10-48f8-be05-63950123ab4f/image.png)


해당 그래프는 어탠션 헤드의 수와 크기에 따른 정확도 변화를 보여준다. x축은 헤드 수이고 y는 정확도를 의미하며 헤드의 크기는 8과 16으로 나누어져 있다.

헤드의 크기가 8일 때는 헤드 수가 2개일 때 최고 성능을 보이며 크기가 16일 때는 헤드 수가 1개일 때 최고 성능을 보인다. 또한 두 가지 경우 모두 최고 성능을 보인 이후부터는 점차 감소하는 추세를 보인다.

이를 통해 **어탠션 헤드의 크기와 수를 줄이는 것이 더 나은 성능을 보임을 알 수 있다**.

![](https://velog.velcdn.com/images/adsky0309/post/d50299e6-5e4d-46ee-8d69-3eb4e2726a0e/image.png)


해당 표는 다양한 어탠션 매커니즘이 정확도에 미치는 영향을 비교한 것이다.

어탠션 매커니즘을 전혀 사용하지 않을 떄의 정확도가 가장 낮았고, 크기가 8인 다중주의 매커니즘(Multi-head self Attention, MSA)를 사용하였을 때 가장 높은 정확도와 k-score를 기록했다.

### G. Comparison to Recent Studies

최근의 연구와 비교한 결과를 살펴보자.

![](https://velog.velcdn.com/images/adsky0309/post/e5aa7748-0321-430d-8d6b-46c5f591d732/image.png)


논문에서 제안하고 있는 ATCNet 모델과 나머지 3가지 모델을 비교하고 있다. 전반적으로 **제안된 모델(ATCNet)이 다른 모델들보다 높은 평균 정확도(85.4%)와 k-점수(0.81)을 보여주고 있다**.

![](https://velog.velcdn.com/images/adsky0309/post/2464e13f-3219-4eee-b4c2-dff6fd0f9e98/image.png)


해당 그래프는 4개 모델의 평균 혼동 행렬을 보여준다. 각 행렬은 Left hand, RIght hand, Foot, Tongue의 4가지 작업에 대한 예측 결과를 나타내는데 색이 밝을 수록 높은 값을 의미하며 대각선에 위치한 요소들이 밝을 수록 해당 작업의 분류 정확도가 높다는 것을 의미한다.

![](https://velog.velcdn.com/images/adsky0309/post/54a00a24-395e-4987-a663-14e30546e7ab/image.png)


해당 표는 BCI-2a 데이터셋에 대한 여러 방법들의 성능을 비교한다. 데이터는 50% 훈련 시행과 50%의 테스트 시행으로 나누어졌으며 가장 하단에 있는 ATCNet 모델이 85.38%의 정확도로 가장 높은 성능을 보여준다.

![](https://velog.velcdn.com/images/adsky0309/post/eca44754-f199-4d78-baf9-62b03244ae9e/image.png)


해당 표는 LOSO(Leave-One-Subject-Out) 교차 검증을 사용한 성능을 보여주며 여기서도 ATCNet 모델이 가장 우수한 성능을 나타낸다.


>📎 LOSO 교차검증
피험자 독립적인 모델 성능을 평가하는데 사용되는 방법으로 모델이 새로운, 보지 못한 피험자의 데이터에 대해 얼마나 잘 일반화되는지를 평가한다.
이를 통해 과적합을 줄이고 모델의 일반화 능력을 더 정확하게 평가할 수 있다.

<br>

## 4. Conclusion

해당 논문은 EEG-MI 분류를 위해 새로운 ATCNet 모델을 제안한다. ATCNet 모델은 **원시 MI-EEG 신호를 압축된 시간 순서로 인코딩하는 CV 블록**, **시간 시퀀스에서 가장 효과적인 정보를 강조하는 멀티헤드 셀프 어탠션 블록**, 그리고 **고수준 시간적 특징을 추출하는 TC블록**으로 이루어져 있다.

각 블록은 ATCNet 모델의 성능에서 중요한 기여를 했으며 비교적 적은 매개변수(115.2k)를 통해 얻어진 결과라는 점에서 제한된 자원을 가지고 있는 산업용 장치에 적용이 가능하다.

제안된 모델은 EEG 신호에서 인위적인 제거 없이 최소한의 전처리만으로도 MI 특성을 뽑아낼 수 있음을 증명하였다.

이후, ATCNet 모델은 여러 영역에서 어탠션 메커니즘을 사용하여 향상시킬 수 있을 것이며 향후 연구 방향으로는 다중 도메인 어탠션 매커니즘 개발, 전처리 방법 개선, 데이터 증강을 위한 생성 모델 사용 등이 있다.

<br>

+ 멘토햄이 논문 리뷰를 할 때는 그림 하나에 있는 숫자들도 다 이해해야 한다고 했다. 이를 테면 나는 합성곱 블록들이 어째서 저런 형태가 되었는지, 그래서 총 몇 초를 해석하고 있는지 물어보면 전혀 알지 못했다. 아키텍쳐 설계를 위한 논문 해석은 참 어려운 것 같다..