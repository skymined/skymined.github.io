---
tags:
  - study/RL
created: 2026-03-26 12:04
변동가능성: true
---
> 개념만 보면 이해가 힘들어서 코드를 통해 이해보고자 한다.
> 개념에 대한 내용은 언젠가 정리...

# Policy Iteration vs Value Iteration

| **항목** | **Policy Iteration**        | **Value Iteration**                    |
| ------ | --------------------------- | -------------------------------------- |
| 특징     | 정책을 고정해 충분히 평가한 뒤 개선        | 매 반복마다 바로 최적 Bellman backup 적용         |
| 정책의 역할 | 명시적 정책이 항상 존재               | 값 함수가 먼저 수렴하고, 마지막에 greedy policy 추출   |
| 장점     | 적은 outer loop로 안정화되는 경우가 많다 | 구현이 짧고 직관적이며 최적성 식과 직접 연결된다            |
| 주의점    | 한 번의 평가가 비쌀 수 있다            | gamma가 크고 stochastic하면 반복 수가 많이 늘 수 있다 |

요약하자면
- Policy Iteration : 정책을 충분히 평가하고 난 다음에 한 번 크게 개선
- Value Iteration: 매번 작은 최적화 백업을 넣어 값을 조금씩 밀어 올린다


# Cho-Mini Development
## 데이터 구조
OpenAI Gym 스타일의 tabular transition model을 단순화해서 사용해본다. 상태 s와 행동 a를 주면 가능한 전이들의 리스트가 나오는 구조다. 완전 심플한 구조이기 때문에 개념 이해를 위한 Mini project로 보면 좋을 것 같음!

```python
P[s][a] = [(prob, next_state, reward, done), ...]  
  
# 예시  
P[0][1] = [(1.0, 1, 0.0, False)]
```
그 외에 사용하는 파라미터는
- gamma: 할인율. 미래 보상을 얼마나 중시할 것인가를 나타내는 것으로 0~1 사이의 값으로 나타낸다
- theta: 반복을 멈출 최소 변화량 기준

## 1. Tiny 3-state MDP
먼저 작은 MDP를 만들어보자. 상태는 A, B, T(terminal)이고 행동은 0과 1 두 개 뿐인 아주 작은 친구.

| **상태** | **행동 0**           | **행동 1**           |
| ------ | ------------------ | ------------------ |
| A(0)   | A로 남음, 보상 -0.2     | B로 이동, 보상 0.0      |
| B(1)   | T로 종료, 보상 +1.0     | A로 이동, 보상 -0.1     |
| T(2)   | 종료 self-loop, 보상 0 | 종료 self-loop, 보상 0 |
검증용 정책은 policy=[1,0,0]으로 둔다. 각 step 별로 선택하는 행동을 의미한다.

```python
def one_step_lookahead(P, s, V, gamma):  
    q = np.zeros(len(P[s]))  
    for a in range(len(P[s])):  
        total = 0.0  
        for prob, next_state, reward, done in P[s][a]:  
            target = reward if done else reward + gamma * V[next_state]  
            total += prob * target  
        q[a] = total  
    return q
```
여기서 나와야 하는 결과값은 해당 정책을 사용했을 때 각 state의 value값이며, gamma=0.9를 사용하면 평가 결과는 V_pi = [0.9, 1.0, 0.0]이 나와야 한다.
`target = reward if done else reward + gamma * V[next_state]` 이 부분은 bellman equation을 코드로 나타낸 것이다. 

## 2. Policy Evaluation
Policy Iteration을 하기 전, 먼저 Policy Evaluation과 Improvement를 살펴보자.
Policy Evaluation은 정책을 고정한 채 값을 반복적으로 갱신한다.
구현 포인트는 두 가지로, 현재 반복에서 계산한 새 값을 바로 쓰지 않도록 new_V를 따로 두는 것, 그리고 최대 변화량 delta가 theta보다 작아지면 종료하는 것이다. 
```python
def policy_evaluation(P, policy, gamma=0.99, theta=1e-8):  
    V = np.zeros(len(P))  
    while True:  
        delta = 0.0  
        new_V = V.copy()  
        for s in range(len(P)):  # 모든 상태를 순회하면서 각 상태의 값을 하나씩 업데이트
            a = policy[s]  # 주어진 policy, state에서의 a 결과값
            value = 0.0  
            for prob, next_state, reward, done in P[s][a]:  
                target = reward if done else reward + gamma * V[next_state]  
                value += prob * target  
            new_V[s] = value  # 해당 state의 value를 가지고 온다.
            delta = max(delta, abs(new_V[s] - V[s]))  
        V = new_V  
        if delta < theta:  
            break  
    return V
```
 Policy evaluation은 모든 상태 s에 대해서 $V^\pi(s)$를 구하는 과정이다. 
 코드에서 보면 알 수 있지만, 모든 상태를 업데이트한 이후에, 변화량의 최댓값을 보고, 계속 계산할지 결정한다. 

예를 들어
```python
V = [0.0, 1.0, 0.0]  
new_V = [0.9, 1.0, 0.0]
```
이 경우 변화량의 max는 0.9이기 때문에 다시 돌려야 한다.

```python
V = [0.9, 1.0, 0.0]  
new_V = [0.9, 1.0, 0.0]
```
이 경우 변화량의 max는 0이 되기 때문에 종료가 가능하다.


## 3. Policy Iteration
Policy Iteration은 평가와 개선을 반복적으로 수행한다. 현재 정책을 충분히 평가한 뒤(2번에서 진행한 것), 각 상태에서 greedy action을 골라 새 정책을 만들고 정책이 더 이상 바뀌지 않으면 종료한다. 이때 1번과 2번에서 만든 함수를 이용해준다.
```python
def policy_iteration(P, gamma=0.99, theta=1e-8):  
    policy = np.zeros(len(P), dtype=int)  # policy는 전부 0으로 초기화
    while True:  
        V = policy_evaluation(P, policy, gamma, theta)  # 2번에서 구한 Policy evaluation 함수
        stable = True  
        for s in range(len(P)):  
            old_action = policy[s]  
            q = one_step_lookahead(P, s, V, gamma)  # 1번에서 구한 q값 구하는 함수
            new_action = np.argmax(q) # q 값이 가장 높은 action을 policy에 넣음
            policy[s] = new_action  
            if new_action != old_action:  
                stable = False  
        if stable:  
            return policy, V
```



## 4. Value Iteration
Policy evaluation을 끝까지 다 하지 않고 곧바로 최적 Bellman Backup을 적용하는 방식이다. 먼저 최적의 V\*을 구하고 난 다음에, 그걸 이용해서 최적의 policy를 구한다.

```python
def value_iteration(P, gamma=0.99, theta=1e-8):  
    V = np.zeros(len(P))  
    while True:  
        delta = 0.0  
        new_V = V.copy()  
        for s in range(len(P)):  # 한 state에서 가능한 모든 a의 q를 계산
            q = one_step_lookahead(P, s, V, gamma)  
            new_V[s] = np.max(q)  
            delta = max(delta, abs(new_V[s] - V[s]))  
        V = new_V  
        if delta < theta:  
            break  
    return V  
  
def extract_greedy_policy(P, V, gamma): 
    policy = np.zeros(len(P), dtype=int)  
    for s in range(len(P)):  
        policy[s] = np.argmax(one_step_lookahead(P, s, V, gamma))  # max q를 만드는 
    return policy
```

`extract_greedy_policy`에서 value function은 이미 구해졌고, 각 상태에서 가장 좋은 action을 `one_step_lookahead`와  `argmax`를 이용해 구해준다.


# Grid World Development