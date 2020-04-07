# ResNet 구축하기 (011 ~ 012)
[![초보 딥러닝 강의-011 ResNet: ResBlock & Pixel Shuffle (1부)](https://i.ytimg.com/vi/drAN7gLA8sU/sddefault.jpg)](https://www.youtube.com/watch?v=drAN7gLA8sU)

[![초보 딥러닝 강의-012 ResNet: Image Regression & Colab Pretrained Results (2부)](https://i.ytimg.com/vi/eSYoOwk31mM/sddefault.jpg)](https://www.youtube.com/watch?v=eSYoOwk31mM)


## Denoising

    python  train.py \
            --mode train \
            --network unet \
            --learning_type residual \
            --task denoising \
            --opts random 30.0
            
    python  train.py \
            --mode train \
            --network resnet \
            --learning_type residual \
            --task denoising \
            --opts random 30.0


## Inpainting

    python  train.py \
            --mode train \
            --network unet \
            --learning_type residual \
            --task inpainting \
            --opts random 0.5
---

    python  train.py \
            --mode train \
            --network resnet \
            --learning_type residual \
            --task inpainting \
            --opts random 0.5


## Super resolution

    python  train.py \
            --mode train \
            --network unet \
            --learning_type residual \
            --task super_resolution \
            --opts bilinear 4.0
            
    python  train.py \
            --mode train \
            --network resnet \
            --learning_type residual \
            --task super_resolution \
            --opts bilinear 4.0
            
    python  train.py \
            --mode train \
            --network srresnet \
            --learning_type residual \
            --task super_resolution \
            --opts bilinear 4.0 0.0