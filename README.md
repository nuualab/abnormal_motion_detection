# abnormal_motion_detection
사람의 쓰러짐 여부를 판별할 수 있는 모델을 배포합니다.   
Efficientnet, Resnest Pretrained 모델에 자체 데이터셋을 추가 학습시켜 성능을 향상 시킨 모델 입니다.   
전이학습(Transfer Learning) 시 사용된 모델의 이름은 아래와 같습니다.

|     모델    |       이름       |
|------------|----------------|
| Efficientnet | efficientnet_b4b |  
| Resnest | resnest50 | 


## Requirements
Python 3.6+.  
Pytorch 1.4.0+


## How to install

```
git clone https://github.com/nuualab/abnormal_motion_detection
```


## How to Train
```
python fall_classification_with_resnest/train/train_fall_classification_with_resnest.py   
python fall_classification_with_efficientnet/train/train_fall_classification_with_efficientnet.py

```


## How to Inference
```
python fall_classification_with_resnest/predict/predict_fall_classification_with_resnest.py --inputdir test_img --device 0
python fall_classification_with_efficientnet/predict/predict_fall_classification_with_efficientnet.py --inputdir test_img --device 0
```
input 디렉토리에 이미지를 넣고 inference를 하게 되면 결과 파일(output.txt)이 아래와 같은 포맷으로 생성 됩니다.   
test1.png 1   
test2.png 0


## Pretrained Model Download

[Efficientnet Pretrained weights](https://drive.google.com/file/d/1oZAZSS0ZYNIn1wsNF-B66csRFCqwnR0N/view?usp=sharing, "Efficientnet")
[Resnest Pretrained weights](https://drive.google.com/file/d/12LjvNFXF6G0QoCQApiGrhdYUEIeqzG7K/view?usp=sharing, "Resnest")


## License
이 프로젝트는 Apache 2.0 라이선스를 따릅니다. 모델 및 코드를 사용할 경우 라이선스 내용을 준수해주세요. 라이선스 전문은 LICENSE 파일에서 확인하실 수 있습니다.

*이 프로젝트는 과학기술정보통신부 인공지능산업원천기술개발사업의 지원을 통해 제작 되었습니다.*
