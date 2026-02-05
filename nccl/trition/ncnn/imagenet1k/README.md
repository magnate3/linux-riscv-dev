

# json
```
 apt install libjsoncpp-dev
```

# opencv

```
 g++ readall.cpp  -o readall -I/usr/local/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
```


# int8 quant


```
./bench_resnet 
total jpeg: 1000
right : 692 wrong: 308  accuracy : 0.692
```


#  imagenet1k 

    - [ImageNet1k](https://www.image-net.org)
    > <img src="/readme_supply/imagenet1k_dataset.png" width=30% height=30%></img>
    - [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)
    > <img src="/readme_supply/cifar10_dataset.png" width=20% height=20%></img>
```
curl -o plane.jpg http://images.cocodataset.org/test2017/000000030207.jpg
curl -o food.jpg http://images.cocodataset.org/test2017/000000228503.jpg
curl -o sport.jpg http://images.cocodataset.org/test2017/000000133861.jpg
curl -o dog.jpg https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg
```

+ ncnn 自带的imagenet-sample-images
```
find ./ -name n02395406_hog.JPEG
./imagenet-sample-images/n02395406_hog.JPEG
```

> ## 测试 n02395406_hog.JPEG

```
/pytorch/ncnn/build/onnx2ncnn/test/build# ./resnet  ../n02395406_hog.JPEG 
output size: 1000
detection time: 102 ms
341 = 5.296703
342 = 4.192142
296 = 3.810430
```
最大是

```
341 = 5.296703
```

和get_label_data_from_filename获取的结果一致 code, label, name =('n02395406', 341, 'hog')

+  运行get_label_data_from_filename函数

```
def get_label_data_from_filename(filename: str, dataset_name: str) -> Tuple[str, int, str]:
    """
    Return name from filename, or (name, label) regarding a name to label dictionnary

    Args:
        filename (str): filename
        datapath (str): path to the dataset

    Returns:
        str|Tuple[str, int, str]: (code, label, name)
    """
    if dataset_name == DATASET_3["name"] or dataset_name == DATASET_1["name"]:
        file_number = get_file_number_from_filename(filename, dataset_name)
        label_idx = imagebet1K_val_groundtruth_labels.get(file_number, None)
        label_code = imagenet1K_labels_to_codes.get(label_idx, None)
        label_name = imagenet1K_labels_to_names.get(label_idx, None)
    else:
        label_code = get_label_code_from_filename(filename, dataset_name)
        label_idx = imagenet1K_codes_to_labels.get(label_code, None)
        label_name = imagenet1K_labels_to_names.get(label_idx, None)
```


```
python3 datasets.py 
Expected n04380533, but got n04380533 for ILSVRC2012_val_00000026_n04380533.JPEG in imagenet_val_images (50K images)
Expected n03710193, but got n03710193 for ILSVRC2012_val_00000152_n03710193.JPEG in imagenet_val_images (50K images)
Expected n02089078, but got n02089078 for n02089078_black-and-tan_coonhound.JPEG in imagenet-sample-images (1K images)
Expected n02395406, but got n02395406 for n02395406_hog.JPEG in imagenet-sample-images (1K images)
-- Tests of get_label_code_from_filename() passed
root@ubuntu:/pytorch/ncnn/build/onnx2ncnn/test/imagenet1k# vim datasets.py 
root@ubuntu:/pytorch/ncnn/build/onnx2ncnn/test/imagenet1k# python3 datasets.py 
Expected n04380533, but got n04380533 for ILSVRC2012_val_00000026_n04380533.JPEG in imagenet_val_images (50K images)
Expected n03710193, but got n03710193 for ILSVRC2012_val_00000152_n03710193.JPEG in imagenet_val_images (50K images)
Expected n02089078, but got n02089078 for n02089078_black-and-tan_coonhound.JPEG in imagenet-sample-images (1K images)
Expected n02395406, but got n02395406 for n02395406_hog.JPEG in imagenet-sample-images (1K images)
-- Tests of get_label_code_from_filename() passed
 code, label, name =('n02395406', 341, 'hog')
```

> ## C++ match

```
#include <iostream>
#include <string>
#include <regex>

int main() {
    std::string text = "2023-10-05";
    std::regex pattern(R"((\d{4})-(\d{2})-(\d{2}))"); // 定义分组
    std::smatch matches;

    // 类似 re.match(pattern, text)
    if (std::regex_match(text, matches, pattern)) {
        // matches[0] 是整个匹配的字符串
        std::cout << "Full match: " << matches[0] << std::endl;
        
        // matches[1], [2], [3] 是各个分组
        std::cout << "Year: " << matches[1] << std::endl;
        std::cout << "Month: " << matches[2] << std::endl;
        std::cout << "Day: " << matches[3] << std::endl;
    } else {
        std::cout << "Match failed" << std::endl;
    }

    return 0;
}
```