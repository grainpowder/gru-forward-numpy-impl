# Boosting computation speed of tensorflow GRU cell

GRU를 포함한 RNN 계열의 모델들은 한 텍스트에 대한 벡터 표현을 얻기 위해, 여러 개의 토큰으로 토큰화시킨 텍스트에 대한 연산을 순차적으로 수행한다. 구체적으로, 각 단계의 연산은 이전 단계까지 계산한 벡터 표현과 현재 단계에서 입력받은 토큰의 임베딩 벡터를 각각 선형 변환한 다음, 변환된 두 벡터들을 가지고 다음 단계에 사용할 벡터 표현을 계산하는 과정으로 이루어져 있다. 이 두 종류의 벡터를 어떻게 변환할지를 정의하는 행렬을 많은 데이터를 사용해서 학습시키게 된다.

학습이 끝나면 토큰 별 임베딩과 이를 변환시킬 행렬은 상수 취급할 수 있다. 이에 따라, 앞서 설명한 단계별 연산에서 이전 단계까지의 벡터 표현이 어떻게 계산될지는 알 수 없지만, 현재 입력받은 토큰의 임베딩이 어떻게 변환될지는 미리 계산해둘 수 있다. 그래서 모든 토큰에 대한 선형 변환 결과를 메모리에 저장해 둔다면, 매 단계마다 토큰의 임베딩을 변환하는 대신 저장된 변환 결과를 참조하기만 하면 되기 때문에 연산량을 줄일 수 있다. 추론 결과를 대용량으로 실시간 서빙해야 하는 상황에서는 이와 같은 최적화가 어플리케이션의 성능을 향상시킬 수 있기 때문에, 메모리 여유가 좀 있는 상황이라면 이 방법을 시도해볼 만 하다. 이 아이디어를 평가할 수 있는 데모를 구현했다.

## Demo setup

* [여기에서](https://www.kaggle.com/nulldata/medium-post-titles) `zip`형태의 데이터를 다운받아 프로젝트 내부의 `data`폴더에 옮겨 놓아야 한다(압축해제 불필요).
* 3.7 버전 이상의 Python이 필요하다.

### on local machine

아래와 같은 커맨드를 입력해 데모를 실행하기 위한 환경을 설정한다(MacOS 기준).

```shell
cd gru-forward-numpy-impl
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
```

### on docker

```shell
cd gru-forward-numpy-impl
docker build -t npgru_image:0.1.1 .
docker run \
  -i -t \
  --rm \
  --name npgru \
  npgru_image:0.1.1
```

# Commentary

다운받은 데이터는 [미디엄에](https://medium.com/) 올라온 약 12만개 포스트들의 제목과 등록 카테고리를 담고 있다. 데모에는 이 데이터를 사용해 토크나이저와 GRU 모델을 학습시키고, 글의 제목을 입력하면 이 제목이 등록될 카테고리를 예측하는 프로세스를 구현했다. 물론 의미있는 예측 결과를 반환하는 모델을 학습시키기에는 다소 부족하지만, 핵심은 Tensorflow 모델을 그대로 서빙에 사용하는 것보다 앞서 설명한 대로 GRU 연산을 재구현해 사용하는 것이 연산 속도가 더 빠름을 보이는 거라서 이 데이터를 사용했다.

## Required Environment Variables

아래와 같은 환경변수를 정의한 `.env`파일을 `src`폴더에 저장한다. 

* FILE_NAME : 다운받은 zip파일에 담긴 파일의 이름(확장자 포함). 
    * 현재는 `medium_post_titles.csv`지만, 혹시 파일명이 변경될 수도 있어서 파일명 하드코딩을 피했음
* ZIPFILE_NAME : zip파일의 이름(확장자 포함)
    * 현재는 `archive.zip`지만, 위와 같은 이유로 파일명을 환경변수를 통해 별도로 입력받게 함 

## Preprocess

`data` 디렉토리에 저장한 압축파일의 압축을 풀어 데이터를 로드하고, 제목에 포함된 대문자를 소문자로 바꾸고 특수문자를 제거한 뒤 카테고리 이름을 인덱스로 변환해 학습, 검증, 평가 데이터(8:1:1 비율)로 분할해 저장하는 프로세스다. 

```shell
python src/npgru/main.py preprocess
```

## Train

전체 데이터의 80%에 해당하는 학습 데이터를 사용해 토크나이저와 토큰 별 임베딩을 학습시키고, 이로부터 제목이 주어지면 등록 카테고리를 예측하는 GRU 모델 기반 분류기를 학습시키는 프로세스다. 학습 진행상황을 확인하기 위해 한 epoch이 종료될 때마다 검증 데이터에 대한 분류 정확도를 계산해 출력하도록 했다. 학습에 사용되는 파라미터를 아래와 같이 지정할 수 있다. 괄호 안에는 디폴트 값을 적었다.
* vocab_size : 토큰 개수(15000)
* embed_dim : 임베딩 벡터 차원 수(128)
* num_epochs : epoch 수(3)
* batch_size : 배치 크기(128)
* num_predict : top-k-precision의 k(3)

```shell
python src/npgru/main.py train \
  --vocab-size=15000 \
  --embed-dim=128 \
  --num-epochs=3 \
  --batch-size=128 \
  --num-predict=3
```

## Evaluate

위 과정에서 학습된 선형 변환에서 사용되는 행렬들을 추출하고 `vocab_size`개의 임베딩 벡터에 대한 변환 결과를 전부 저장한 뒤 GRU 모델의 forward 계산 로직을 `numpy`로 재구현한 클래스의 인스턴스를 생성한다. 이 인스턴스의 추론 속도와 tensorflow 모델의 추론 속도를 비교한다. 

```shell
python src/npgru/main.py evaluate
```

추론에 걸리는 시간은 문장의 길이에 비례해서 길어지기 때문에, 여러 길이의 제목에 대한 추론 시간을 측정해서 비교했다. 제목 별 추론시간(밀리초)의 중위값을 계산해 비교한 예시를 적었다. `numpy`로 재구현한 클래스의 추론 속도가 기존 모델의 추론 속도보다 항상 몇 배는 빠른 것을 알 수 있다. 위 커맨드를 실행했을 때 출력되는 로그에서도 비슷한 결과를 확인할 수 있다.

```
Length | tf elapse(ms) | np elapse(ms)
--------------------------------------
     2 |        3.8478 |        0.2971
    26 |         7.539 |        1.0333
    50 |        9.8625 |        1.5662
    74 |       12.9421 |        2.0611
    99 |        15.663 |         2.847
```

## Upload (optional)

머신에서 동기적으로 실험을 진행해서 얻은 결과만으로는 서빙 환경에서도 `numpy`로 재구현한 클래스가 더 우수하다고 단정지을 수는 없다. 직접 API를 만들어서 실제 서빙환경에서 이 둘을 비교해야 완전한 비교가 마무리된다. 이를 위해서는 적합시킨 모델을 별도의 저장소에 업로드하는 프로세스가 필요하다. 이 데모에서는 AWS S3 저장소에 결과 파일들을 업로드하는데, 이를 위해서는 아래의 환경변수를 `.env`파일에 추가한다.

* AWS_ACCESS_KEY_ID : AWS 액세스 키 ID
* AWS_SECRET_ACCESS_KEY :  AWS 액세스 키 비밀번호
* S3_BUCKET_NAME : S3 버킷 이름

```shell
python src/npgru/main.py upload
```