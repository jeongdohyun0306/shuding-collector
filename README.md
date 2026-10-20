# 슈딩이 컬렉터 v2

학교 고양이 슈딩이 사진을 한국어 문장으로 찾는 검색기의 개선판이다. 원본 발표(`Introducing Schding Collector.pdf`)의 방식(v1)을 그대로 재현해 두고, 같은 정답표로 개선판(v2)과 비교했다.

## 무엇이 바뀌었나

v1은 사진과 검색어를 EmbeddingGemma 2로 같은 768차원 공간에 넣고 코사인 유사도로 순위를 매긴다. 원본의 검색어 전처리는 내용을 알 수 없어서 "슈딩이"를 "고양이"로 바꾸는 것만 넣었다(v2와 같은 이름 처리).

v2는 원본이 이미 쓰던 두 모델(EmbeddingGemma 2, Qwen3-VL-2B-Instruct)만 쓴다. 검색할 때 GPU에 올라가는 건 v1과 같은 EmbeddingGemma 2 하나다.

1. 색인할 때 Qwen3-VL-2B가 사진마다 3문장 설명을 쓰고, 그 설명도 EmbeddingGemma 2로 임베딩한다. 검색 점수는 사진 유사도와 설명 유사도를 각각 표준화해서 더한다.
2. 검색어에서 부정 조건(안, 못, ~지 않, ~지 못하, ~지 말고, 없이, 없는, 아닌, 말고, 빼고)을 떼어 낸다(`sc/kparse.py`). 한국어 형태소 분석기 Kiwi로 형태소 단위에서 찾기 때문에 "안경안쓴", "사람없이"처럼 붙여 써도 잡고, "상자 안", "편안하게", "쉴 새 없이"는 부정으로 읽지 않는다. "안경을 안 쓴 슈딩이"는 찾을 것 "고양이", 뺄 것 "안경을 착용한 고양이"로 나뉜다.
3. 뺄 것과 많이 닮은 사진은 점수를 깎는다. 뺄 것에 대한 사진 점수와 설명 점수를 각각 표준화해 더한 값이 1을 넘으면, 넘은 만큼 뺀다. 이 기준(1)과 감점 크기는 dev 검색어로 골랐다.

## 결과 (nDCG@10, 높을수록 좋음)

| 검색어 묶음 | 개수 | v1 | v2 | 차이 95% 신뢰구간 |
|---|---|---|---|---|
| 처음 보는 검색어 (2차 held-out) | 44 | 0.486 | 0.685 | +0.136 ~ +0.265 |
| 그중 부정 조건 | 35 | 0.461 | 0.679 | |
| 벤치마크 전체 | 47 | 0.735 | 0.862 | +0.078 ~ +0.182 |
| 벤치마크 단일 조건 | 25 | 0.850 | 0.879 | +0.004 ~ +0.054 |
| 벤치마크 복합 조건 | 10 | 0.721 | 0.792 | +0.022 ~ +0.122 |
| 벤치마크 부정 조건 | 12 | 0.505 | 0.886 | +0.279 ~ +0.471 |

부정 조건 검색에서 상위 10장 중 금지 조건을 어긴 사진 비율은 벤치마크에서 24.2%에서 2.5%로, 처음 보는 검색어에서 28.0%에서 9.7%로 줄었다. 질의 시간 중앙값은 단일·복합 조건에서 v1과 같고(약 36 ms), 무언가를 빼는 검색에서는 v2가 임베딩을 두 번 하고 Kiwi도 돌려서 37 ms에서 79 ms로 늘었다(RTX 5060 노트북, 2026-10-20 재실행. 처음 잰 값은 45 ms와 83 ms였다).

### 고정 이후 분해기 수정 (2026-10-20)

위 표의 처음 보는 검색어 0.685는 분해기를 고정한 판(커밋 0b8196e)의 결과다. 그 44개에서 v2가 v1보다 크게 진 5개 중 3개가 분해기 범위 오류였고, 이를 고쳤다.

- 관계절 뒤의 "사진/모습 빼고·말고": "사람 품이나 무릎에 있는 사진 빼고"에서 "사진"만 빼던 것을 관계절까지 뺀다.
- 나열: "사람이나 다른 고양이, 강아지 없이"에서 마지막 "강아지"만 빼던 것을 셋 다 뺀다. 쉼표는 양쪽 명사가 같은 부류(사람·동물, 착용품, 장소)일 때만 나열로 본다.
- "다른 고양이는 안 보이는": "다른 고양이"가 부정 범위에서 빠지던 것을 고쳐 "다른 고양이와 함께 있는 고양이"를 뺀다.
- 같은 계열 오류로, "말고 다른 데서"의 "다른"이 찾을 문장에 남던 것도 지운다.

고친 판으로 44개를 다시 채점하면 0.758(v1 대비 +0.272, 39개 나아짐, 1개 나빠짐), 금지 조건 위반 3.4%다. 다만 점수가 바뀐 검색어 6개는 전부 고칠 때 들여다본 것이고 나머지 38개는 점수가 소수점까지 같다. 그래서 이 수치는 처음 보는 검색어 결과로 볼 수 없다. 벤치마크 47개는 분해 결과가 하나도 바뀌지 않아 점수가 같다. 결과는 `results/heldout2_after_fix.json`, 분해기 회귀 시험은 `tests/test_kparse.py`에 있다. "밖에서 찍은 사진 빼고"처럼 "에서"가 붙은 관계절은 아직 "밖에서"를 범위에 넣지 못한다.

조건 종류별·부정 형태별·실사진 13장 분석은 `bench/breakdown.py`(결과 `results/breakdown.json`, 분해기를 고치기 전 판으로 냈다), 그 주장들의 독립 반박 검증은 `bench/audit/claims_verification.json`에 있다. 2026-10-20에 실사진 2장의 "사람" 라벨 누락을 고쳤고, 고치기 전 결과는 `results/*_before_label_fix.json`이다.

정답 라벨은 사람이 아니라 Claude(AI)가 사진을 한 장씩 보고 붙였다. 다른 Claude 에이전트가 101장을 따로 라벨링해서 대조한 코드 단위 일치율은 98.1%다. 벤치마크 검색어와 부정 분해기를 같은 작성자(Claude)가 만들었기 때문에 벤치마크 수치는 부풀려져 있을 수 있다. 처음 보는 검색어 줄이 더 믿을 만한 숫자다. 이 검색어는 분해기 코드를 고정(`bench/FROZEN_ENGINE_SHA256.txt`)한 뒤 코드를 본 적 없는 에이전트 셋이 썼고, 다른 에이전트가 문장 뜻과 정답식이 맞는지 검사해서 56개 중 12개를 뺐다.

## 실행

```
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe "torch==2.14.1+cu130" torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/Scripts/python.exe transformers sentence-transformers accelerate qwen-vl-utils pillow pillow-heif numpy fastapi uvicorn pymupdf pandas fastparquet kiwipiepy
```

torch를 설치한 뒤 다른 패키지를 `-U`로 올리면 CPU판 torch로 덮어써지니 주의.

슈딩이 사진 폴더 색인 (사진당 약 2초, 대부분 설명 생성 시간, 아이폰 HEIC 가능):

```
.venv/Scripts/python.exe -m sc.build --photos "D:/슈딩이 사진" --out index_shuding
```

검색 화면:

```
.venv/Scripts/python.exe app/server.py --index index_shuding
```

브라우저에서 `http://127.0.0.1:8765` 를 연다. 오른쪽 위에서 개선판, 원래 방식, 비교를 고른다. 결과 썸네일을 누르거나 좌우 화살표로 넘기면 해당 사진으로 이동한다. 고양이 이름은 `app/config.json`의 `names`에서 바꾼다.

## 평가 재현

COCO 고양이 사진 288장은 저장소에 넣지 않았다. 아래 명령이 `data/coco_meta/chosen.json`에 적힌 주소에서 같은 사진을 받아 `data/coco_photos/`에 둔다. 사진 임베딩과 설명(`index/`)은 저장소에 들어 있어서, 사진을 받으면 GPU로 다시 색인하지 않아도 채점할 수 있다.

```
.venv/Scripts/python.exe tools/fetch_coco.py
.venv/Scripts/python.exe bench/report_all.py
.venv/Scripts/python.exe bench/heldout2.py bench/audit/heldout2_workflow.json
.venv/Scripts/python.exe bench/agreement.py bench/audit/blind_labels.json
.venv/Scripts/python.exe tests/test_kparse.py
```

`report_all.py`는 벤치마크, 버린 구성 11개, 감점 격자, 1차 held-out, 논란 라벨 제외, 벡터 차원 줄이기, 질의 시간을 한 번에 다시 내고 `results/summary.json`에 쓴다. `heldout2.py`는 지금 코드의 결과를 `results/heldout2_after_fix.json`에 쓴다. 고정 판의 `results/heldout2.json`을 다시 내려면 `git checkout 0b8196e -- sc/kparse.py`로 분해기를 되돌린 뒤 마지막 인자로 `heldout2.json`을 준다.

## 폴더

- `sc/` 검색 엔진. `v1.py` 원래 방식, `v2.py` 개선판, `kparse.py` 부정 분해, `vlm.py` Qwen3-VL-2B 설명·번역·확인, `rerank.py`·`verify.py` 버린 구성 재현용, `build.py` 폴더 색인
- `bench/` 검색어(`queries.json`, `heldout2_queries.json`)와 채점. 고정 해시는 `FROZEN_SHA256.txt`, `FROZEN_ENGINE_SHA256.txt`. `audit/`에 독립 검증 원자료
- `data/` 평가용 사진 301장 (슈딩이 13장, COCO 2017 고양이 288장)과 정답 라벨 `labels_raw.txt`
- `app/` 검색 화면
- `results/` 채점 결과
- `tools/` COCO 사진 표본 뽑기·내려받기, 라벨링용 사진 시트 만들기
- `tests/` 부정 분해기 회귀 시험

## 출처

- 원래 방식(v1)과 문제 설정은 발표 자료 「슈딩이 컬렉터」를 따랐다. 원래 앱의 코드는 이 저장소에 없다.
- `data/real_photos/`의 슈딩이 사진 13장은 그 발표 자료에서 꺼냈다. 사진 저작권은 발표 자료에 적힌 촬영자에게 있다.
- COCO 2017 캡션과 사진 목록: COCO Consortium, CC BY 4.0. 사진 자체는 Flickr 이용약관을 따르므로 저장소에 넣지 않았다.
- 모델: [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2), [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) (둘 다 Apache 2.0). 버린 구성 재현에만 Qwen3-VL-Embedding-2B, Qwen3-VL-Reranker-2B를 썼다.
- 한국어 형태소 분석: [Kiwi](https://github.com/bab2min/kiwipiepy) (kiwipiepy, LGPL v3).
