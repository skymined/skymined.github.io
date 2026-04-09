---
tags:
  - paper/ca
created: 2026-04-09 16:04
변동가능성: false
---
- 논문 링크: [https://doi.org/10.1145/3197517.3201311](https://doi.org/10.1145/3197517.3201311)  
- 프로젝트 페이지: [https://xbpeng.github.io/projects/DeepMimic/index.html](https://xbpeng.github.io/projects/DeepMimic/index.html)


## 먼저 알아야 할 것

DeepMimic의 핵심은 아주 간단하게 말하면 "<font color="#ffc000">**모션 클립처럼 보이게 움직이되, 실제 물리 시뮬레이션 안에서 버티고 적응하고 목표도 수행하는 정책**</font>"을 강화학습으로 학습하는 것이다. 입력으로는 캐릭터의 물리 모델, 하나 이상의 기준 동작 클립, 그리고 필요하다면 목표 방향이나 타격 목표점 같은 태스크 목표가 들어간다. 정책은 현재 몸 상태와 모션의 진행 정도를 나타내는 phase, 그리고 목표 정보를 받아 각 관절의 PD 타깃 자세를 출력한다. 물리 엔진은 그 타깃을 바탕으로 실제 토크를 계산하고, 그 결과로 몸이 움직인다.

중요한 점은 DeepMimic이 단순히 "모션을 복제"하려는 것이 아니라는 데 있다. 보상은 크게 두 덩어리로 나뉜다. 하나는 <font color="#ffc000">기준 모션과 얼마나 닮았는지를 보는 imitation reward</font>이고, 다른 하나는 <font color="#ffc000">목표를 얼마나 잘 수행했는지를 보는 task reward</font>다. 이 둘을 함께 최적화하면, 정책은 원래 모션의 스타일을 유지하면서도 상황에 맞게 조금씩 동작을 바꿀 수 있다. 이 구조 덕분에 걷기, 달리기, 점프, 스핀킥, 백플립 같은 고난도 동작뿐 아니라, 목표 방향으로 달리기, 발로 타깃 치기, 공 던지기, 장애물 지형 통과 같은 상호작용형 동작도 가능해진다.

이 논문에서 정말 중요한 것은 단순히 PPO를 썼다는 사실이 아니다. DeepMimic의 실질적인 성패를 가르는 요소는 <font color="#ffc000">reference state initialization(RSI)</font>과 <font color="#ffc000">early termination(ET)</font>이다. 논문은 이 두 가지가 특히 공중 동작이나 고난도 접촉 동작을 배우는 데 결정적이라고 실험으로 보여준다. 그래서 **DeepMimic을 이해할 때는 '보상 함수'만이 아니라 '에피소드를 어떻게 시작하고 언제 끊는가'까지 포함해서 학습 문제를 설계했다는 점**을 봐야 한다.

## 1. Introduction

서론에서 저자들은 물리 기반 캐릭터 애니메이션의 오랜 목표를 아주 명확하게 잡는다. 현실적인 물리 반응을 하면서도, 모션 캡처나 애니메이터가 만든 고품질 움직임의 스타일을 그대로 유지하고 싶다는 것이다. 기존의 키네마틱 기반 애니메이션은 모션 품질이 매우 좋지만, 외부 힘이나 환경 변화에 대한 반응이 약하다. 반대로 순수 강화학습 기반 제어는 일반성은 높지만 움직임이 어색해지기 쉽다. 논문이 겨냥하는 문제는 바로 이 둘의 간극이다.

저자들이 제시하는 아이디어는 복잡한 제어 구조를 새로 만드는 것이 아니라, 강화학습 보상 안에 "기준 모션과 닮을수록 보상을 준다"는 항을 직접 넣는 것이다. 그리고 여기에 목표 달성 보상을 추가한다. 이 접근은 겉으로 보면 단순하지만, 실제로는 중요한 전환점이다. 기존 많은 시스템은 키네마틱 모션 생성기 위에 물리 추적 컨트롤러를 얹는 식이었다. 그런데 이런 구조는 "추적 가능한 모션"을 먼저 만들어야 하고, 정책이 상황에 따라 모션을 과감하게 바꾸기도 어렵다. DeepMimic은 아예 정책이 직접 물리 세계 안에서 움직이며 기준 모션을 닮도록 학습하게 만든다.

>[!note] Kinematic motion 생성기는 뭐고 물리 추적 컨트롤러는 무엇인가?
> - Kinematic motion 생성기는 어떻게 움직이면 모양이 자연스러운가를 만드는 모듈이다. 현재 자세와 사용자의 명령을 보고 앞으로의 목표 포즈를 예측한다. 이때 중요한 점은 torque와 같은 직접적인 힘을 계산하는게 아니라 그냥 사람다운 모션을 제안한다는 점이다.
> - 물리 추적 컨트롤러는 Kinematic motion이 제안한 모션을 실제 물리 시뮬레이터 안에서 따라가게 해주는 모듈이다. 이쪽이 직접 토크를 내거나 PD target을 내는 것. 
> 	- PD(Proportional-Derivatvie) target: 어떤 값을 목표값으로 맞추기 위한 가장 기본적인 피드백 제어기
> 	- 예를 들어 지금 관절 각도를 목표 각도로 맞추고 싶다고 가정하자. 지금 목표보다 많이 벗어나 있으면 강하게 당기고 너무 빠르면 덜 밀거나 감쇠를 주는 식이다.
> 	- e.g. 토크 = kp \* (목표각도 - 현재각도) + kd * (목표속도 - 현재속도 )

서론에서 또 하나 꼭 봐야 하는 부분은 논문의 기여를 어떻게 정의하는가이다. 저자들은 자신들이 완전히 새로운 한 개의 알고리즘을 발명했다고 주장하지 않는다. 대신 이미 알려진 여러 구성 요소들, 즉 policy gradient, imitation reward, phase-aware policy, PD target action, PPO, 적절한 초기 상태 분포와 종료 조건을 데이터 기반 캐릭터 제어에 맞게 결합해, 이전보다 훨씬 넓은 범위의 기술을 실제로 학습시켰다는 점을 기여로 둔다. 조합을 잘 했다는 이야기. 

## 2. Related Work
관련 연구는 DeepMimic이 어느 축 위에 서 있는지 보여준다. 첫째 축은 **키네마틱 기반 방법**이다. 이 계열은 충분한 데이터가 있으면 매우 자연스러운 움직임을 만들 수 있다. 하지만 새로운 지형, 새로운 외란, 새로운 태스크가 들어오면 대응 범위가 제한된다. 결국 엄청나게 많은 모션 데이터를 모아야 하고, 데이터에 없는 상황으로 일반화하는 능력이 약하다. DeepMimic은 여기에 물리 시뮬레이션을 넣어 "상황이 바뀌면 몸도 물리적으로 바뀌어야 한다"는 prior를 활용한다.

둘째 축은 **물리 기반 최적화와 모델 기반 제어**다. 이쪽은 걷기 같은 특정 기술에는 굉장히 강력했지만, 접촉이 복잡하거나 장기 계획이 필요한 고난도 동작, 또는 매우 다양한 기술을 하나의 프레임워크로 처리하는 데는 한계가 있었다. DeepMimic은 모델 프리 RL로 더 일반적인 틀을 유지하면서, 사람 손으로 구조를 세세히 설계하지 않고도 다양한 기술을 얻는 방향을 택한다.

셋째 축은 **순수 강화학습과 Motion Imitation**이다. 기존 deep RL은 연속 제어 문제를 풀 수 있었지만, 보상 설계가 부정확하면 팔을 쓸데없이 흔들거나, 이상한 보폭으로 달리거나, 몸이 비틀리는 등 보기 어색한 움직임을 자주 만들었다. (논문에서는 이를 *artifact* 아티펙트라고 부른다.) 모션 캡처 기반 imitation이나 GAIL류 접근도 있었지만, 당시 결과는 여전히 전통적인 애니메이션 품질에는 못 미쳤다. DeepMimic은 여기서 adversarial imitation 대신, 상태 유사도를 직접 측정하는 imitation reward를 사용한다. **즉, 판별기를 따로 배우지 않고 "얼마나 닮았는가"를 손으로 정의한 보상으로 주는 쪽이다.**

관련 연구 중에서 저자들이 특히 비교 대상으로 두는 것은 **SAMCON**이다. SAMCON은 이미 매우 다양한 고난도 기술을 보여준 강력한 시스템이었지만, 구조가 복잡하고, 저차원 제어 구조를 별도로 설계해야 하며, 태스크 목표나 고차원 입력을 자연스럽게 넣기 어렵다. DeepMimic은 SAMCON만큼 혹은 그에 근접한 기술적 난이도의 움직임을, 훨씬 단순한 deep RL 프레임워크로 얻고자 한다. 이 비교가 중요하다. DeepMimic은 "최초의 모션 모방"이 아니라 "실전성 있는 단순한 RL 프레임워크"라는 위치에 있다.


## 3. Overview
개요 파트는 시스템 전체를 한 문장으로 묶는다. 입력은 캐릭터 모델, 기준 모션들, 그리고 보상 함수로 정의된 태스크다. 출력은 기준 모션을 닮으면서도 태스크를 수행할 수 있는 제어 정책이다. 여기서 기준 모션은 시간에 따라 변하는 target pose 시퀀스 $\hat{q}_t$로 표현된다. 정책 $\pi(a_t \mid s_t, g_t)$는 현재 상태 $s_t$와 목표 $g_t$를 받아 행동 $a_t$를 내보내고, 이 행동은 각 관절의 PD 컨트롤러가 따라야 할 목표 각도로 해석된다.

즉 DeepMimic은 trajectory optimizer가 아니다. **매번 미래 전체 궤적을 다시 푸는 것이 아니라, 현재 상태를 보고 바로 관절 타깃을 출력하는 reactive policy를 학습**한다. 그래서 **학습은 오래 걸리지만, 한 번 학습된 뒤에는 실행이 빠르고, 외란이나 지형 변화에도 즉시 반응**할 수 있다. 이 점이 게임, 인터랙티브 애니메이션, 로봇 제어로 이어질 수 있는 실용적인 장점이다.


## 4. Background
일반적인 강화학습으로, 상태 $s$, 행동 $a$, 보상 $r$, 정책 $\pi$, 그리고 기대 누적 보상 $J(\theta)$를 최대화하는 파라미터 $\theta$를 찾는 것이 목표다. 논문에서 중요한 것은 수식 자체보다, 왜 policy gradient 계열이 여기 맞는가이다. DeepMimic의 행동 공간은 고차원 연속 공간이고, 동작은 물리 시뮬레이터를 거치므로 미분 가능한 닫힌 형태의 제어 해를 구하기 어렵다. 그래서 궤적 샘플을 모아 gradient를 추정하는 방식이 자연스럽다.

또 하나 중요하게 봐야 할 것은 advantage와 value function의 역할이다. value는 "이 상태에서 앞으로 평균적으로 얼마나 잘할 것 같은가"를 추정하고, advantage는 "지금 취한 행동이 그 평균보다 얼마나 좋았는가"를 본다. 따라서 policy gradient는 평균보다 좋았던 행동의 확률은 올리고, 나빴던 행동의 확률은 낮추는 방향으로 간다. DeepMimic은 이 기본 틀 위에 PPO를 얹고, value는 TD로, advantage는 GAE로 계산한다. 각 방식에 대해서는 알아보길!


## 5. Policy Representation
 DeepMimic의 정책이 정확히 무엇을 보고 무엇을 내보내는지를 알아보자. 기준 모션은 단지 "**원하는 자세의 시퀀스**"만 줄 뿐이다. **실제 물리 세계에서는 관성, 충돌, 토크 제한, 지면 접촉 때문에 그 자세를 그대로 따라갈 수 없다**. 따라서 정책이 해야 하는 일은 단순한 자세 복사가 아니라, **현재 몸 상태를 감안해서 다음 순간 어떤 관절 목표를 줘야 비슷한 궤적으로 갈 수 있는지를 계산하는 것**이다.

### 5.1 States and Actions
상태는 루트, 즉 골반을 기준으로 한 각 링크의 상대 위치, 회전, 선속도, 각속도로 구성된다. 중요한 점은 이 모든 특징을 캐릭터의 local coordinate frame에서 계산한다는 것이다. 이렇게 해야 세계(global) 좌표계에서 어디를 보고 있든, 정책은 보다 일관된 입력 분포를 본다. 여기에 모션 진행 정도를 나타내는 phase 변수 $\phi \in [0,1]$가 들어간다. $\phi = 0$은 모션 시작, $\phi = 1$은 모션 끝이고, 주기 동작이면 다시 0으로 돌아간다.**같은 몸 상태라도 지금이 동작 초반인지 후반인지에 따라 취해야 할 제어가 완전히 다르기 때문**이다. 예를 들어 달리기의 왼발 착지 직전과 오른발 착지 직전은 몸의 일부 상태가 비슷해 보여도, 다음에 내야 할 힘은 다르다. phase는 정책이 "지금 기준 모션의 어느 시점에 있어야 하는가"를 알게 해주는 시계 역할을 한다. 

> **해당 방식은 논문의 한계이기도~**
> `phase`는 유용하지만 정책을 몸 상태 기반 제어가 아니라 시간 기반 추적이 되어버리기 때문이다. 지금 0.3일 경우 A자세로 간다고 하자. 그런데 perturbation(외란)이나 환경 변화가 생기면 실제 몸의 진행도가 phase의 진행도와 어긋나게 된다.

행동은 토크 자체가 아니라 **각 관절의 PD 컨트롤러가 추적할 목표 방향**이다. 구면 관절은 axis-angle, 회전 관절은 스칼라 각도로 표현한다. 토크를 직접 출력하게 하면 학습 난이도가 높아지고, 로컬 damping이나 feedback까지 정책이 전부 떠안아야 한다. PD target action은 저수준 제어 일부를 컨트롤러에 위임해서, 정책이 더 상위의 "자세를 어디로 유도할 것인가"에 집중하게 만든다. DeepMimic이 안정적으로 다양한 동작을 배우는 데 이 액션 공간 설계가 크게 기여한다.

### 5.2 Network
정책은 상태와 목표를 받아 Gaussian action distribution을 출력하는 신경망이다. 
$$
\pi(a|s) = N(\mu(s), \Sigma)
$$
Gaussian action distribution은 정책이 행동을 딱 하나의 값으로 내는 게 아니라 행동의 확률분포를 내는 것이다. 여기서 a는 각 관절의 PD Target들이고 $\mu$는 지금 상태에서 가장 그럴 듯한 중심값, $\Sigma$(공분산 행렬)은 그 주변의 퍼짐 정도다. 

> [!note] 왜 가우시안을 사용할까?
> 1. 행동 공간이 연속값이기 때문이다. 
> 2. 탐색이 필요하기 때문이다. 항상 같은 행동만 내면 새로운 동작을 찾을 수 없다.
> 3. PPO와 같은 policy gradient가 쓰기 편하기 때문이다. PPO는 "이 행동이 현재 정책 아래에서 얼마나 그럴듯했는가"를 계산해야 하는데 가우시안은 log를 계산하기 쉬워서 학습이 안정적이다.

평균 $\mu(s)$는 네트워크가 예측하고, 공분산은 대각선 고정값으로 둔다. 기본 구조는 fully connected 1024, 512 두 개의 은닉층과 선형 출력층이다. value network도 거의 같은 구조를 쓰되 출력이 스칼라 하나다. 지금 기준으로 보면 아주 단순한 MLP지만, 당시 이 정도 차원의 상태와 행동 공간을 안정적으로 다루는 것 자체가 쉽지 않았다.

```text
[state, goal] 
   -> Linear(1024) + ReLU
   -> Linear(512) + ReLU
   -> Linear(action_dim)
```

여기서 논문은 지형 관련된 정보를 넣어중어야 한다. 평평한 바닥에서 걷는 태스크는 몸 상태만 알아도 어느 정도 되지만 장애물 회피나 stairs 같은 문제는 바깥 환경도 봐야하기 때문이다. 그래서 지형을 보는 시각 기반 태스크에서는 주변 terrain heightmap을 추가 입력으로 넣고, 이를 convolutional layers로 처리한다.

> [!note] Heightmap이란?
> 주변 바닥 종이를 격자 형태로 적어놓은 지도로 단순 벡터가 아니라 2D grid 같은 공간 구조를 가진다. 때문에 MLP보다 convolutional layer가 더 잘 처리한다.

그런 다음 heightmap feature를 일반 상태와 목표 feature에 붙여서 다시 MLP로 보낸다. 즉, DeepMimic은 "몸 내부 상태만 보는 controller"에서 멈추지 않고, 외부 환경을 perception input으로 받을 수 있는 visuomotor policy까지 확장된다.

![[Pasted image 20260409164922.png]]

### 5.3 Reward
DeepMimic의 전체 보상은 아래처럼 imitation과 task의 선형 결합이다.

$$
r_t = \omega_I r_t^I + \omega_G r_t^G
$$

여기서 $r_t^I$는 기준 모션을 얼마나 잘 따라갔는지, $r_t^G$는 태스크를 얼마나 잘 수행했는지를 본다. 태스크가 없는 순수 imitation 실험에서는 사실상 $r_t^I$가 전부다. 태스크가 있는 경우 논문은 실험에서 보통 $\omega_I = 0.7, \omega_G = 0.3$을 사용한다. 이 수치는 DeepMimic 철학을 잘 보여준다. 우선순위는 여전히 "모션 스타일 유지"이고, 그 안에서 목표를 수행하도록 하는 것.

imitation reward는 다시 네 가지 항으로 분해된다.
$$
r_t^I = 0.65\,r_t^p + 0.10\,r_t^v + 0.15\,r_t^e + 0.10\,r_t^c
$$

첫째는 **pose reward**다. 각 관절의 회전이 기준 자세와 얼마나 가까운지를 quaternion 차이로 본다. 둘째는 **velocity reward**다. 관절 각속도가 기준 모션의 속도와 얼마나 비슷한지를 본다. 셋째는 **end-effector reward**로, 손과 발의 위치를 맞춘다. 넷째는 **center-of-mass reward**로, 전체 질량중심의 위치를 맞춘다. 각 항은 오차 제곱에 음수를 곱한 뒤 지수함수를 취하는 형태다. 그래서 오차가 작으면 보상이 1에 가깝고, 오차가 커지면 빠르게 줄어든다. DeepMimic은 "정확한 joint-by-joint tracking" 하나만 강제하지 않는다. **관절 자세, 관절 속도, 손발의 공간적 위치, 몸 전체의 중심 이동을 함께 보게 함으로써, 보기 좋은 움직임의 여러 측면을 동시에 압축**한다. pose만 맞추면 손발 타이밍이 어긋날 수 있고, end-effector만 맞추면 몸통이 망가질 수 있다. COM 항이 없으면 전체 무게중심 이동이 부자연스러워질 수 있다. 따라서 이 네 항의 조합이 DeepMimic 움직임의 자연스러움을 만든다.

동시에 이 보상은 한계도 만든다. 논문 마지막에서 저자들이 인정하듯, 이 imitation metric은 결국 사람이 손으로 설계한 상태 유사도다. 어떤 움직임을 원하는지는 사람이 정하지 않았지만, 그 움직임을 잘 따라했는지 측정하는 방식은 사람이 정한다.  즉, "무엇이 자연스러운가"를 완전히 데이터에서 학습한 것이 아니라, 사람이 관절/손발/COM이 중요하다고 판단해 넣은 것이다. DeepMimic은 end-to-end처럼 보이지만, 실제로는 꽤 강한 inductive bias가 들어간다.(고 생각한다.)

> [!important] DeepMimic는 데이터로 목표 모션을 주지만, 그 데이터를 reward로 해석하는 방식은 손으로 직접 설계했다.

## 6. Training
학습은 **PPO 기반의 episodic training**으로 진행된다. 각 에피소드 시작 시 초기 상태를 reference motion에서 샘플링하고, 정책이 행동을 샘플링하며 rollout을 만든다. 일정 길이까지 시뮬레이션하거나 종료 조건에 걸리면 에피소드를 마치고, 모은 데이터로 policy와 value를 업데이트한다. 여기까지는 평범하다. 하지만 DeepMimic에서 중요한 것은 exploration을 "알고리즘 트릭"만으로 해결하지 않고, 에피소드 구조 자체를 바꿔 해결했다는 점이다.

저자들은 강화학습에서 exploration 문제가 단지 intrinsic reward의 문제가 아니라, 에피소드를 어디서 시작하고 언제 자르느냐의 문제이기도 하다고 본다. 그리고 이 시각이 DeepMimic의 고난도 동작 학습 성공에 거의 결정적이다.

### 6.1 Initial State Distribution
보통 RL 문제는 고정된 initial state에서 시작한다. 하지만 모션 imitation에서는 이 선택이 매우 나쁘게 작동할 수 있다. 예를 들어 백플립을 처음부터 시작 상태에서만 배우면, 정책은 먼저 점프 초반을 배우고, 그 다음 공중 회전, 그 다음 착지 순서로 진척해야 한다. 그런데 실제로는 착지를 할 수 있어야 점프 전체가 좋은 보상을 받는다. 착지를 못하면 높이 뛰는 것 자체가 오히려 손해가 된다. 즉, **나중 단계의 성공이 앞 단계의 탐색을 유도해야 하는데, 고정 시작 상태에서는 그 연결이 너무 약하다**.

이를 해결하는 것이 <font color="#ffc000">RSI, reference state initialization</font>이다. 에피소드 시작 시 reference motion의 아무 시점이나 샘플링해서 그 상태로 캐릭터를 초기화한다. 그러면 정책은 학습 초반부터 모션의 중간이나 후반에 해당하는 "좋은 상태들"을 직접 만나게 된다. 백플립이라면 공중 회전 중간 상태, 착지 직전 상태 같은 것들을 처음부터 본다. 저자들이 말하는 핵심은, reference motion이 단지 reward를 통해서만 정보를 주는 것이 아니라, initial state distribution을 통해서도 정보를 준다는 것이다. 이것은 exploration 관점에서 엄청나게 강한 힌트다.

DeepMimic을 재현하거나 확장할 때 많은 사람이 보상식에만 집착하는데, 실제로는 RSI가 없으면 특히 flight phase가 긴 동작을 거의 배우지 못할 가능성이 높다. 논문의 ablation도 이 점을 정면으로 보여준다.

### 6.2 Early Termination
**early termination(ET)은 캐릭터가 특정 실패 상태에 들어가면 에피소드를 바로 끊는 것**이다. 여기서는 몸통이나 머리 같은 특정 링크가 바닥에 닿으면 종료한다. 그리고 남은 시간 동안의 보상은 0으로 본다. 이 역시 단순한 구현 세부사항처럼 보이지만, 논문은 이것이 학습 품질에 매우 큰 영향을 준다고 보여준다.

이유는 두 가지다. **첫째, ET는 보상 shaping 역할**을 한다. 넘어져 바닥을 구르는 상태를 강하게 불리하게 만들어, 정책이 그런 local optimum에 머무르지 못하게 한다. **둘째, 데이터 분포를 정제**한다. ET가 없으면 학습 초기에 에피소드 대부분이 "넘어진 뒤 바닥에서 허우적거리는 상태"로 가득 차게 된다. 그러면 네트워크 용량 상당 부분이 본래 배우고 싶은 유용한 동작이 아니라 실패 상태를 모델링하는 데 쓰인다. 

즉, ET는 실패 상태를 과감하게 잘라내서, 정책이 성공 가능성이 있는 상태들에 더 많은 학습 용량을 쓰도록 만드는 핵심 장치다.

## 7. Multi-Skill Integration
하나의 모션만 잘 따라 하는 정책은 인상적이지만, **실제 캐릭터 제어에서 더 중요한 것은 여러 기술을 연결하고 상황에 따라 선택하는 능력**이다. DeepMimic은 이를 위해 세 가지 방식을 제안한다.

**첫 번째는 multi-clip reward**다. 여러 reference clip 각각에 대해 imitation reward를 계산하고, 그 순간 가장 큰 값을 주는 clip의 reward를 사용한다. 
$$
r_t^I = \max_{j=1, ..., k} r_t^j
$$
이 방식은 비슷한 계열의 동작들을 묶을 때 유용하다. 예를 들어 직진 걷기와 여러 종류의 회전 걷기 클립을 함께 넣어 두면, 정책은 현재 상황에서 가장 잘 맞는 걸음 패턴을 스스로 고른다. 중요한 것은 별도의 kinematic planner 없이도 clip 전환이 일어난다는 점이다. 다만 논문 결과를 보면 이 방법은 서로 비슷한 기술끼리 묶을 때 잘 되고, frontflip과 sideflip처럼 성격이 다른 기술을 한데 넣으면 일부 클립만 모방하는 문제가 생길 수 있다.

**두 번째는 skill selector**다. goal 입력을 one-hot vector로 두고, 어떤 skill을 지금 실행할지 사용자가 지정하게 한다. 학습 중에는 매 cycle 시작 때 무작위 skill을 고르게 해서, 정책이 여러 기술과 그 사이 전환을 함께 배우게 만든다. 이 방식은 하나의 네트워크 안에 여러 기술을 넣되, 사용자가 명시적으로 "지금은 이 동작"을 고르길 원할 때 적합하다.

세 번째는 composite policy다. 이는 개인적으로 DeepMimic에서 가장 인상적인 아이디어 중 하나다. 여러 단일-skill 정책을 각각 따로 학습해 두고, 실행 시점에 각 정책의 value function을 보고 지금 상태에서 어떤 정책이 가장 유망한지를 정한다. 구체적으로는 각 정책의 value에 대해 softmax, 즉 Boltzmann 분포를 만들고, 그 확률로 sub-policy를 선택한다. 값이 높은 정책은 "지금 이 상태에서 내가 이어받아도 잘할 수 있다"는 뜻이므로 더 자주 선택된다. 이 방식의 장점은 새로운 조합을 만들기 위해 매번 멀티스킬 정책을 다시 학습하지 않아도 된다는 것이다. 논문은 넘어진 후 get-up policy가 자동으로 선택되는 사례를 보여주는데, 이건 사실상 value가 transition feasibility estimator로 쓰인다는 뜻이다.

정리하면, multi-clip reward는 비슷한 스타일의 연속체를 하나로 묶는 데 좋고, skill selector는 사용자 명령형 인터페이스에 좋고, composite policy는 이미 학습된 다양한 기술을 라이브러리처럼 조합하는 데 좋다. DeepMimic은 멀티스킬 문제를 하나의 정답으로 밀지 않고, 상황별 도구 상자를 제시한다.

## 8. Characters

논문은 단일 인간형 캐릭터만 다루지 않는다. humanoid, Atlas, T-Rex, dragon까지 네 종류의 캐릭터를 사용한다. 모두 articulated rigid body로 모델링되며, 대부분의 관절은 3자유도 spherical joint이고 팔꿈치와 무릎만 1자유도 revolute joint다. 각 관절에는 PD 컨트롤러가 있고, gain은 캐릭터마다 사람이 정해서 고정한다.

이 파트에서 중요한 것은 "다양한 morphology"를 진짜로 다뤘다는 점이다. Atlas는 humanoid와 비슷한 구조지만 질량 분포와 actuator 특성이 크게 다르고, T-Rex와 dragon은 아예 사람 모션 캡처가 없는 비인간 캐릭터다. 논문은 이들에게 keyframed animation을 기준 모션으로 넣어서도 학습이 가능하다고 보여준다. 즉 DeepMimic은 mocap 전용 시스템이 아니라, "어떤 reference motion 표현이든 물리 정책으로 번역하는 프레임워크"에 가깝다.

또 하나 봐야 할 부분은 차원 수다. humanoid는 상태 197차원, 행동 36차원이고, dragon은 상태 418차원, 행동 94차원이다. 이는 전통적인 continuous control benchmark보다 훨씬 큰 수준이다. 그래서 DeepMimic의 성과는 단순히 "재미있는 애니메이션"이 아니라, 고차원 연속 제어를 실제로 다루는 RL 사례로도 의미가 있다.

## 9. Tasks

DeepMimic이 단순 motion tracking을 넘어서려면, imitation 외에 추가 태스크를 넣을 수 있어야 한다. 이 파트는 그 goal-conditioned formulation을 보여준다.

첫 번째 태스크는 target heading이다. 캐릭터가 특정 수평 방향으로 움직이도록 하는 목표다. 보상은 목표 방향 성분의 속도가 원하는 속도보다 느릴 때만 패널티를 준다. 즉, 목표보다 너무 빠르다고 벌주지는 않는다. 이 설계는 자연스럽다. 걷거나 달리는 동작을 유지하면서 목표 방향으로 충분히 전진하게 만들고, 약간 더 빠른 것은 허용한다.

두 번째는 strike다. 캐릭터의 발이나 손 같은 특정 링크로 무작위 위치의 구형 타깃을 치게 한다. 흥미로운 점은 goal 입력에 타깃 위치뿐 아니라, 이미 타깃을 쳤는지 여부를 나타내는 이진 변수 \(h\)를 넣는다는 것이다. 정책이 feedforward network라 내부 메모리가 없으니, 외부에서 최소한의 memory bit를 상태로 주는 셈이다. 이건 당시 recurrent policy를 피하면서도 필요한 상태 전이를 다루는 실용적인 선택이다.

세 번째는 throw다. 구조는 strike와 비슷하지만, 타깃을 직접 치는 대신 공을 던져 맞혀야 한다. 그래서 상태에 공의 위치, 회전, 선속도, 각속도가 추가된다. 나중 결과 파트에서 보듯, imitation 없이 task reward만 주면 정책은 공을 던지지 않고 공을 들고 목표 쪽으로 달려가는 기괴하지만 기능적인 전략을 택한다. 이 사례는 DeepMimic에서 imitation reward가 왜 중요한지 아주 잘 보여준다. task reward만으로는 "성공"은 할 수 있어도 "그럴듯한 동작"은 나오지 않는다.

네 번째는 terrain traversal이다. mixed obstacles, dense gaps, winding balance beam, stairs 같은 지형을 통과해야 한다. 여기서 DeepMimic은 heightmap 기반 visuomotor policy를 사용한다. 특히 학습을 빠르게 하기 위해 먼저 flat terrain에서 일반 MLP로 기본 모션을 학습한 뒤, 그 네트워크에 heightmap과 convolution layer를 붙여 복잡한 지형으로 옮겨 간다. 이 progressive learning은 처음부터 perception-heavy setting에서 모든 것을 동시에 배우는 것보다 훨씬 실용적이다.

## 10. Results

결과 파트는 DeepMimic이 "정말 되는가"를 다각도로 입증한다. 정책은 30Hz로 실행되고, 물리 시뮬레이션은 Bullet에서 1.2kHz로 돌아간다. 신경망은 TensorFlow로 학습했다. 사람형 캐릭터는 걷기, 달리기, 춤, 크롤링, 점프, 발차기, 구르기, 백플립, 프론트플립, 스핀킥, 보울트 등 매우 다양한 기술을 학습했고, Atlas와 비인간 캐릭터에도 적용됐다.

표 2를 보면 쉬운 주기 동작일수록 정규화 return이 높고, 공중 회전이나 긴 접촉을 포함하는 동작은 상대적으로 낮다. 예를 들어 humanoid walk는 0.985, run은 0.951, jump는 0.947로 높지만, frontflip은 0.485, landing은 0.590, spin은 0.664다. 이 숫자를 볼 때는 주의가 필요하다. 논문도 말하듯 normalized return의 최대값 자체가 실제로 도달 불가능한 경우가 있다. 따라서 낮은 수치가 항상 "실패"를 뜻하지는 않는다. 중요한 것은 정성적으로도 정책이 기준 모션과 유사한 동작을 안정적으로 재현했다는 점이다.

### 10.1 Tasks

태스크 실험의 핵심 질문은 이것이다. "정말 imitation과 task를 함께 최적화하면, 스타일은 유지하면서 목표도 더 잘 달성하는가?" 표 4의 답은 그렇다. spinkick strike 태스크에서 imitation과 task를 함께 쓴 정책은 99% 성공률을 보였고, imitation만 쓴 정책은 19%에 그쳤다. baseball pitch throw에서는 둘 다 쓴 정책이 75%, imitation만 쓴 정책이 5%였다. 즉 기준 모션을 그대로 따라 하는 것만으로는 목표를 잘 달성할 수 없다. 정책은 필요할 때 기준 모션에서 벗어나야 한다.

반대로 task reward만 주고 imitation을 빼면 목표는 수행할 수 있지만 동작이 어색해진다. 공 던지기 태스크에서 정책은 공을 던지기보다 공을 손에 든 채 목표로 달려간다. 이것은 DeepMimic의 가장 중요한 메시지 중 하나다. "task-only RL은 기능적이지만 보기 싫고, imitation-only는 보기 좋지만 목적성이 약하다. 둘을 함께 넣어야 비로소 usable motion이 된다."

### 10.2 Multi-Skill Integration

멀티스킬 실험은 7장에서 제안한 세 방법이 실제로 작동하는지 보여준다. multi-clip reward로 걷기와 회전 클립 다섯 개를 함께 넣고 목표 heading을 따라가게 하면, heading이 바뀔 때 turning clip들이 활성화되고, 다시 정렬되면 forward walk가 주로 선택된다. 이 결과는 max-over-clips reward가 실제로 상황에 따라 다른 reference를 활용하도록 정책을 이끈다는 증거다.

skill selector에서는 flip 묶음과 jump 묶음을 하나의 정책 안에서 배우고, one-hot 입력으로 임의 순서의 기술을 실시간으로 실행할 수 있음을 보인다. composite policy에서는 backflip, frontflip, sideflip, cartwheel, spinkick, roll, 그리고 get-up 정책들을 따로 학습한 뒤 value 기반으로 합친다. 여기서 특히 중요한 장면은 캐릭터가 넘어지면 별도의 스크립트 없이 적절한 get-up 정책이 선택된다는 점이다. 즉 value function이 "이 상태에서 어떤 기술로 이어가야 살아남는가"를 판단하는 스위치로 작동한다.

### 10.3 Retargeting

retargeting 실험은 DeepMimic의 일반성을 강하게 뒷받침한다. 첫째는 character retargeting이다. humanoid 기준 모션의 local joint rotation을 거의 그대로 Atlas에 복사하고, Atlas 전용 정책을 새로 학습하면 걷기, 달리기, 백플립, 스핀킥이 다시 나온다. 반대로 humanoid에서 학습한 정책을 Atlas에 그대로 적용하면 거의 아무 것도 되지 않는다. 이 비교는 아주 중요하다. DeepMimic은 "정책 자체가 morphology-invariant"하다는 주장이 아니라, "같은 reference motion style을 다른 morphology의 물리 모델에도 다시 학습시킬 수 있다"는 주장이다.

둘째는 environment retargeting이다. 평지 착지 모션을 기준으로 삼았는데, 실제 학습은 2m 높이의 ledge에서 내려오게 할 수 있다. 또 단일 달리기 클립으로 장애물, 연속 gap, winding balance beam, irregular stairs를 통과하는 정책을 만들 수 있다. 이는 reference motion이 환경과 완전히 같아야 한다는 뜻이 아니라, 기준 스타일을 주는 seed로 작동하고 정책이 환경에 맞게 변형할 수 있음을 의미한다.

셋째는 physics retargeting이다. 논문은 moon gravity에서도 spinkick과 cartwheel을 학습시킨다. 중력장이 달라져도 모션 스타일을 어느 정도 유지하며 적응할 수 있다는 뜻이다. 이는 DeepMimic이 단순 playback이 아니라, 실제 동역학 아래에서 동작을 다시 조직하는 policy learning이라는 점을 잘 보여준다.

### 10.4 Ablations

이 논문에서 반드시 봐야 하는 실험이다. 저자들은 RSI와 ET를 빼면 무슨 일이 일어나는지를 정량적으로 비교한다. 결론은 매우 분명하다. early termination은 많은 기술에서 결정적이고, RSI는 특히 공중 체공 구간이 긴 기술에서 결정적이다.

표 5를 보면 backflip의 정규화 return은 full method(RSI+ET)에서 0.791인데, RSI를 빼면 0.379까지 떨어진다. sideflip도 0.823에서 0.355로 크게 감소한다. 반면 walk는 0.980에서 0.974 정도로 큰 차이가 없다. 이 비교가 중요한 이유는, RSI와 ET의 효과가 쉬운 locomotion보다 고난도 acrobatics에서 훨씬 크다는 것을 보여주기 때문이다. 즉 DeepMimic의 성공은 보상 함수 하나로 설명되지 않는다. 학습 과정의 curriculum-like shaping, 특히 어디서 시작하고 언제 실패로 판정할지의 설계가 핵심이다.

저자들이 지적하듯, RSI 없이 backflip을 학습시키면 return 수치만 얼핏 비슷해 보여도 실제 동작을 보면 제대로 공중 회전을 하지 못하고 뒤로 짧게 홉만 뛰는 식의 실패가 나온다. 이 대목은 정량 지표만 보면 놓치기 쉽다. DeepMimic에서는 motion quality 평가가 반드시 정성 검토와 함께 가야 한다.

### 10.5 Robustness

강건성 평가는 학습된 정책에 외부 힘을 가해 어디까지 버티는지 본다. 골반에 0.2초 동안 힘을 가하고, 넘어진다면 그 직전 크기를 최대 허용치로 기록한다. humanoid run은 전방 720N × 0.2s까지, spinkick은 전방 690N × 0.2s와 측면 600N × 0.2s까지 버틴다. 이는 당시 SAMCON과 비슷하거나 더 나은 수준이라고 저자들은 말한다.

특히 흥미로운 점은 이런 외란을 학습 중에 명시적으로 주지 않았다는 것이다. 저자들은 stochastic policy의 exploration noise가 결과적으로 강건성에 기여했을 것이라고 추정한다. 이건 DeepMimic을 로봇 쪽으로 읽을 때 중요한 힌트다. 모든 강건성을 domain randomization이나 disturbance training으로 직접 주입하지 않아도, 적절한 stochastic training과 imitation objective의 결합만으로 상당한 회복 능력이 생길 수 있다는 뜻이다.

## 11. Discussion and Limitations

논문은 결과를 과장하지 않고 한계를 솔직하게 적는다. 첫째, phase variable이 기준 모션과 선형 동기화되어 있으므로, 동작의 타이밍을 늘이거나 줄이는 적응이 어렵다. 외란을 받았을 때 자연스럽게 템포를 늦췄다가 다시 맞추는 식의 유연성이 제한된다. 이 한계는 이후 phase-functioned network나 latent timing control, adversarial motion prior 계열 연구들이 다루게 되는 지점과도 연결된다.

둘째, multi-clip integration은 소수의 클립에는 잘 동작하지만, 대규모 모션 라이브러리까지 확장되었다고 말하기는 어렵다. 셋째, 저수준 PD controller의 gain과 torque limit은 여전히 사람이 적절히 잡아 줘야 한다. 넷째, 학습 비용이 크다. supplementary에 따르면 humanoid 단일 스킬 하나를 학습하는 데 대략 6천만 샘플, 8코어 CPU 기준 약 2일이 걸린다. 다섯째, imitation reward는 결국 손으로 설계한 유사도이며, imitation과 task의 상대 가중치도 신중하게 맞춰야 한다.

그럼에도 DeepMimic의 의의는 매우 크다. 이 논문은 "모션 품질은 데이터에서, 적응성과 강건성은 물리 기반 강화학습에서" 가져오는 조합이 실제로 통한다는 것을 보여줬다. 그리고 이후 등장한 AMP, adversarial motion prior, motion diffusion prior, humanoid control, sim-to-real locomotion 연구들의 출발점 중 하나가 된다.

## Supplementary Material

supplementary는 본문에서 당연하게 넘어간 RL 세부사항을 보강한다. DeepMimic을 재현하거나 변형하려면 이 부분도 같이 이해하는 편이 좋다.

### A. Multi-Step Returns

Monte Carlo return은 unbiased하지만 분산이 크고, 1-step return은 분산이 작지만 bias가 생긴다. n-step return과 \(\lambda\)-return은 이 둘 사이의 절충이다. DeepMimic은 value 업데이트에 TD(\(\lambda\))를 쓰고, advantage 추정에는 GAE(\(\lambda\))를 쓴다. 직관적으로는 "너무 먼 미래까지 그대로 믿지도 않고, 한 스텝 bootstrap만 고집하지도 않는" 절충형 추정기다. 동작 제어처럼 장기 구조가 중요하지만 샘플 분산이 큰 문제에서는 이 선택이 매우 합리적이다.

### B. Off-Policy Learning

PPO는 완전한 off-policy 알고리즘은 아니지만, 이전 정책 \(\pi_{\theta_{\text{old}}}\)에서 모은 샘플을 importance sampling으로 재사용한다. 비율

\[
w_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
\]

은 현재 정책이 옛 정책에 비해 해당 행동에 얼마나 더 큰 확률을 주는지를 나타낸다. 이 비율 덕분에 한 번 모은 rollout으로 여러 update step을 수행할 수 있어 샘플 효율이 올라간다.

### C. Proximal Policy Optimization

PPO의 핵심은 정책이 한 번의 업데이트에서 너무 멀리 바뀌지 않게 하는 것이다. TRPO가 KL 제약으로 trust region을 만들었다면, PPO는 likelihood ratio를 \([1-\epsilon, 1+\epsilon]\) 범위로 clip하는 더 단순한 surrogate objective를 사용한다. DeepMimic은 이 clipped PPO를 사용한다. 의미는 간단하다. 현재 action이 좋아 보인다고 해도, 그 action의 확률을 한 번에 과도하게 올리면 학습이 불안정해진다. PPO는 이런 폭주를 막아 준다. DeepMimic처럼 고차원 연속 제어에서 정책이 자주 무너질 수 있는 환경에서는 이 안정성이 매우 중요하다.

### D. Learning Algorithm and Hyperparameters

supplementary의 알고리즘 1은 DeepMimic 전체 학습 루프를 요약한다. reference motion에서 초기 상태를 샘플링해 캐릭터를 초기화하고, 4096개 샘플을 모은 뒤, 크기 256의 minibatch로 policy와 value를 업데이트한다. \(\gamma = 0.95\), \(\lambda = 0.95\), PPO clipping threshold는 \(\epsilon = 0.2\)다. value step size는 \(10^{-2}\), policy step size는 humanoid와 Atlas에서 \(5 \times 10^{-5}\), dragon과 T-Rex에서 \(2 \times 10^{-5}\)를 쓴다. optimizer update는 momentum 0.9의 SGD로 수행한다.

이 세부 설정은 재현성 측면에서 중요하다. DeepMimic은 최신 기준으로 보면 엄청나게 큰 네트워크나 복잡한 optimizer를 쓰지 않는다. 오히려 적당히 단순한 네트워크, 명확한 reward, 신중한 episode design, 그리고 PPO 하이퍼파라미터 세팅으로 성능을 만든다. 동시에 학습량은 결코 작지 않다. 단일 스킬에도 수천만 샘플이 필요하고, GPU가 아니라 CPU 기반으로 며칠이 걸렸다는 점을 보면, 이 논문이 "쉽게 되는 트릭"을 말하는 것은 아니라는 점도 분명하다.

## 결국 DeepMimic이 남긴 것

DeepMimic을 한 줄로 정의하면, "reference motion을 물리적으로 실행 가능한 skill prior로 바꾸는 강화학습 프레임워크"다. 모션 캡처는 스타일을 준다. 물리 엔진은 현실성을 준다. imitation reward는 그 스타일을 잃지 않게 붙잡아 둔다. task reward는 목표 지향성을 넣는다. PPO는 이를 최적화한다. RSI와 ET는 고난도 동작도 실제로 배우게 만든다. 그리고 value function은 나중에 멀티스킬 전환의 판단 기준으로까지 쓰인다.

따라서 DeepMimic을 제대로 이해했다는 것은 단순히 "모션 캡처를 imitation reward로 넣은 PPO"라고 말하는 수준이 아니다. 정말 중요한 이해는 다음과 같다. DeepMimic은 보상 설계, 행동 공간 설계, 상태 표현, 에피소드 시작 분포, 실패 종료 규칙, value 활용 방식이 서로 맞물려 돌아가는 시스템이다. 이 중 하나만 떼어 보면 평범해 보일 수 있지만, 이 조합이 2018년 기준으로는 매우 강력했고, 이후 physics-based humanoid control 연구의 기본 문법이 되었다.>)

