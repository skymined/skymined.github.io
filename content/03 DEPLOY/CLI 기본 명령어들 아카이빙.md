---
tags:
created: 2026-03-02 14:31
변동가능성: true
---
CLI란 Command Line Interface로 마우스 대신 텍스트 명령어로 컴퓨터를 조작하는 방법이다. 우리가 평소에 쓰는 건 GUI(그래픽 버튼을 누르는 것)이고 CLI는 보통 아래와 같이 생겼다.
```bash
ls  
cd my_project  
python train.py
```

내가 매번 잊어버리기도 하고 기본적인 명령어들은 아카이빙 되어 있는 것이 필요해서 정리해두려고 한다.


## 1 파일 & 디렉토리 관련
**\[파일/디렉토리 탐색 및 이동]**
파일 시스템을 탐색하고 현재 위치를 확인하는 명령어들이며, `ls` 과 `pwd` 같이 디렉토리의 내용을 확인하거나 경로를 표시하는 도구들이 포함
```bash
ls       # 디렉토리 내용 나열 (옵션 -a 숨김파일, -l 상세 정보)
pwd      # 현재 작업 중인 디렉토리의 절대 경로 표시:contentReference[oaicite:2]{index=2}
cd       # 다른 디렉토리로 이동
 tree    # 현재 디렉토리의 트리 구조 표시
 locate  # 파일 이름으로 위치 검색:contentReference[oaicite:3]{index=3}
 find    # 조건에 따라 파일/디렉토리를 검색:contentReference[oaicite:4]{index=4}
```

파일 내용 확인 및 페이징
파일의 내용을 화면에 출력하거나 일부만 보여주는 명령이다. 긴 텍스트를 읽을 때 페이지 단위로 보여주는 `less`/`more` 등도 포함
