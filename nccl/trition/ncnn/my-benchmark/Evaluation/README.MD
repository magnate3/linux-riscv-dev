# Evaluation for ncnn models

This is a repository to evaluate ncnn models performance on different datasets.

## File organization
```
evaluation/
 |_ src/
 |_ ncnn/
 |_ models/
 |_ build/
```

## How to build (on Ubuntu)
1. fetch code
```
git clone https://github.com/DiamondSheep/Evaluation.git
cd Evaluation
```
2. install required libs
```
sudo apt update 
sudo apt install libboost-dev 
```

3. install [ncnn](https://github.com/Tencent/ncnn) and [OpenCV](https://github.com/opencv/opencv.git) (follow their install tutorial)

4. prepare ncnn model and data
```
mkdir models
mv ${ncnn_model} ./models/
```

5. build
```
mkdir build
cd build/
cmake .. && make 
./evaluate ${model_name} ${data_path}
```
