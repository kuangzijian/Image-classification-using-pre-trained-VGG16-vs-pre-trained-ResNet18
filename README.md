
<!-- PROJECT LOGO
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Best-README-Template</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
</p>
-->


<!-- TABLE OF CONTENTS 
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)
-->


<!-- ABOUT THE PROJECT -->
## UAlberta-Multimedia-Master-Program-MM811-2020-Assignment-3

This is a school project assigned by University of Alberta Master of Multimedia program. This project focuses on topics:
* Improved Transfer Learning using PyTorch for image classification
* Image classification using pre-trained VGG16 vs pre-trained ResNet18

### Built With
* [Pytorch](https://github.com/pytorch)

### Prerequisites
```sh
1. Clone the repo
2. pip install -r requirements.txt
```

### Improved Transfer Learning algorithm for image classification

```
- To run training/validation on the original CNN
python train.py

- To run training/validation on the improved pre-trained VGG16 model
python train_with_vgg16.py

- To run training/validation on the improved pre-trained ResNet18 model
python train_with_resnet18.py
```
### Training/Validation outputs

```
Model  Training Accuracy  Validation Accuracy
CNN               81.57%               65.26%
VGG16             83.70%               79.62%
ResNet18          96.58%               76.33%
```
After applying pre-trained ResNet18 to our image classifier, the result shows there is overfitting issue:

![Result before applying dropout](https://github.com/kuangzijian/UAlberta-Multimedia-Master-Program-MM811-2020-Assignment-3/blob/main/images/before.png)

So dropout has been applied into the fine-tuning FC layers to avoid overfitting:

![Result after applying dropout](https://github.com/kuangzijian/UAlberta-Multimedia-Master-Program-MM811-2020-Assignment-3/blob/main/images/dropout.png)

The result looks much better after applying dropout, to further solve overfitting issue, data augumentation will be applied in next step.

![Result after applying dropout](https://github.com/kuangzijian/UAlberta-Multimedia-Master-Program-MM811-2020-Assignment-3/blob/main/images/after.png)
<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

## References
Transfer learning using PyTorch https://www.analyticsvidhya.com/blog/2019/10/how-to-master-transfer-learning-using-pytorch/

ResNet18 https://docs.google.com/presentation/d/1bTbZRi3LaziU2Oml7Bs8LIWRRTZ0LUTiSO8hl9nLgXs/edit#slide=id.g9cae5f4108_6_5564

