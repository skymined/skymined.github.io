---
tags:
  - paper/ca
created: 2026-04-09 16:03
변동가능성: true
---
몇가지 논문들을 읽어보고자 한다. 순서는 내가 읽은 순서로 진행하며 개인적으로 나의 생각과 공부 과정이 담겨 있다. 특정 논문을 클릭하면 해당 논문을 자세하게 기록한 문서로 자동 이동하니 참고!

### DeepMimic(TOG 2018)
[[DeepMimic; Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills]]
Physics Animation 연구자 왈 무조건 읽어야 하는, 이 분야에 몸담은 사람들은 무조건 한 번씩은 봤을 논문. 새로운 방법을 제시하기보다 기존에 존재했던 방법론을 잘 조합한 논문이다. 실제 동역학을 학습하지 않아 외부 충격에 약한 Imitation learning과 학습 공간이 넓어 학습이 어렵고 결과물에 artifact가 존재하는 Reinforcement learning을 엮어 외부 충격에도 강건하며 자연스러운 Animation을 만들었다. 해당 논문에서 주목할 점은 State를 무작위로 초기화하는 Initial State Distribution과 특정 실패로 들어가면 바로 학습을 중단하는 Early Termination을 통해 학습 품질을 올렸다는 점이다. 
실제로 모션을 생성하는 논문인 줄 알았는데 motion clip들이 있고 그 clip들을 몇 가지 방법으로 조합해서 이를 따라하도록 설계한 것이었다.

### Learning Predict-and-simulate policies from unorganized human motion data(TOG 2019)
[[Learning Predict-and-simulate policies from unorganized human motion data(2019)]]
일단 DeepMimic의 한계를 생각해보면, 기본적으로 저 친구는 짧고 잘 정리된 reference clip을 따라가는 논문이다. 그래서 한 스킬씩 잘 재현하는 것은 잘하지만 큰 규모의 뒤섞인 모션 데이터에서 여러 행동과 전이를 한 번에 배우기 어렵다. 
그래서 이 논문은 정해진 모션을 잘 따라가는게 아니라 정리되지 않은 무작위 데이터에서 앞으로 무엇을 하면 좋을지 예측하면서 물리적으로 실행하는 법을 보여준다. 사용자가 명령하면 그것에 맞춰 움직일 수 있도록 학습하는 것.


### AMP


### ASE
