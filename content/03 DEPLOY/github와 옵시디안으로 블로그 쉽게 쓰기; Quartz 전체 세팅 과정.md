---
tags:
  - setup
created: 2026-02-28T14:32:00
---
돌고돌아 블로그 페이지 만들기 뽕이 차올라버렸다. 사실 github를 이용하는 게 제일 좋다는 건 어느 정도 알고 있었으나 업로드하는 방식과 마크다운 제작방식이 너무나도 귀찮아서 tistory와 밸로그를 왔다갔다하고 있었다.
그런데 과제 스트레스로 도피성 아이디어가 떠올랐으니, 어차피 같은 마크다운 형식이고 내가 그것에 익숙하다면 옵시디안과 github를 연동해서 이용할 수는 없을까?

>[!note] **결론적으로 만들 수 있다. 매우 쉽게.**

내 옵시디안 폴더가 연구실이라고 하면 `quartz`는 출판사의 개념이고 Githb는 서점인 것.
떄문에 출판사인 quartz에 넘긴 원고(00 BLOG)만 Github에 올라가도록 해보겠다.

## STEP1 Github 저장소 제작

이름은 `<name>.github.io`

## Step2 Quartz 설치
```bash
git clone https://github.com/jackyzha0/quartz.git  
cd quartz  
npm install
```

- 이때 Node.js가 설치되어 있어야 함


> Window Powershell에서 content 삭제하는 법

```bash
rm content -Recurse -Force
```

###  A안. 만약 Vault 안에 00BLOG를 만들 경우우
- powershell에서 진행
- 
```bash
cd C:\Users\sky\Documents\quartz #quartz의 위치
Remove-Item content -Recurse -Force # 그 안에 있는 content 폴더를 지우고
New-Item -ItemType SymbolicLink -Path "C:\Users\sky\Documents\quartz\content" -Target "C:\Users\sky\Documents\skyworld\00 BLOG" # 내가 올리고자 하는 친구를 대신 넣어준다
```

그러면 결과가
```bash
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d----l      2026-02-28   오후 2:17                content
```
이런 식으로 나오는데 d----1은 link가 되었다는 뜻이다. 그리고 Quartz의 content은 이제 내 `00 BLOG`를 가리키고 있다.

이제
```bash
npx quartz build --serve
```
를 입력하고 브라우저에서 `http://localhost:8080`를 입력하면 아래와 같은 내용이 나오는데, 
그 전에 00 BLOG 폴더에 `index.md`를 만들어서 넣어주어야 한다. 그럼 내 블로그가 나옴.

### B안. 그냥 아래와 같은 구조를 만들경우
``` plain text
C:\Users\sky\Documents
 ├── skyworld          ← 비공개 연구 Vault (절대 Git 안 함)
 └── quartz
      ├── content      ← 공개용 글 (여기서 옵시디안 작성)
      ├── package.json
      ├── quartz.config.ts
      └── .github/workflows/deploy.yml
```

이 경우에는 더 쉽다. 그냥 옵시디안 vault 폴더를 quartz와 연동시키면 됨.
나는 결국 이 B안으로 가기로 함.


## 정식 배포를 위한 단계
### Pages 설정 변경
Settings → Pages → Source-> `GitHub Actions`  로 변경

### deploy.yml 파일 추가
```
quartz
 └── .github
      └── workflows
           └── deploy.yml
```
아래 폴더 구조를 만들고
deploy.yml 파일 만들기
```YAML
name: Deploy Quartz site

on:
  push:
    branches:
      - main

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: 20

      - name: Install dependencies
        run: npm ci

      - name: Build Quartz
        run: npx quartz build

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: public

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

### Github에 push
그 전에 quartz 는 원본 템플릿이기 때문에 origin을 내 레포로 바꿔주는 작업을 진행한다.
(아마 지금 `git remote -v`를 하면 quartz.git이 나올 것)

```powershell
git remote set-url origin https://github.com/skymined/skymined.github.io.git
git push -u origin v4
```

앞으로 옵시디안에서 글을 쓴다고 하더라도 quartz 폴더에서 아래와 같은 작업을 진행해주어야 한다.
```powershell
git add .
git commit -m "Update posts"
git push
```

근데 사실 내가 옵시디안이랑 분리해서 하려다 보니까(=전체 Vault와 공개용 사이트를 분리) 그렇게 된 거고, 만약 그렇게 안 해도 된다면 그냥 contents 안에 내용을 넣어도 되지 않을까 싶다.



## BLOG 초기 꾸미기
`quartz.config.ts`  파일에 들어가서 여러가지를 변경하면 됨

- baseURL 변경: `baseUrl: "skymined.github.io`
- 팔레트 변경: `darkmode`라고 되어 있는 곳에 팔레트를 바꾸려고 함. 나는 남색&노랑 계열을 좋아해서 변경

이렇게 완성된 블로그는 몇 가지 불편한 점이 있는데 그 중 하나가 목록이다. 진짜 옵시디안처럼 작동하기 때문에 tistory와 같은 좋은 가시성을 기대하면 안된다. 때문에 나는 여러 가지를 직접 바꿨다.
- 폴더만 왼쪽 목록에 드러나게 설정. 하위 폴더를 선택할 경우 리스트가 나타남
- 상위 폴더와 하위 폴더의 색상 변경
- 상위 폴더는 항상 `OPEN` 상태로 두며 이때 정렬은 알파벳 순

파일 정렬도 제대로 안되어 있기 때문에 옵시디안의 속성을 이용했다.
```YAML
---
tags:
created: 2026-03-01 02:39
---
```
이런 식으로 template을 만들어두면(자세한 건 옵시디안 template 만들기라고 치면 나올 것) 알아서 created 시간이 뜨는데, 이 역시 `quartz.config.ts`에 들어가서 '파일'은 created 순으로 정렬 세팅해두면 된다.

그것 외에는 Linkdin, github, mail 버튼을 만들고 내 사진을 넣을 공간을 만든 정도.

이제 블로그 만들기가 참 쉬워졌다......