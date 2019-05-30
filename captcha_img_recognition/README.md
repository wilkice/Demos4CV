# Implement captcha image recognition with PyTorch 1.1
> Any PR or issue is welcome! :smiley:

This project is mainly focus on building a captcha image recognition model. The train and validate accuracy go to 70% after 30 epochs of training on 1 million images.

The captcha img contains 4 characters from a-z, A-Z, 0-9.
![](https://i.bmp.ovh/imgs/2019/05/cc817892cbfa0b6f.png)

* For loss function, I use `torch.nn.CrossEntropyLoss` to calculate each loss of the 4 character and add them all.
* Grey image is used to reduce computation power and faster train process



### Need to install:
* [captcha](https://github.com/lepture/captcha): generate captcha imgs
`pip install captcha`
* [tdqm](https://github.com/tqdm/tqdm): a Fast, Extensible Progress Bar for Python and CLI
`pip install tqdm` or 
`conda install -c conda-forge tqdm`
* ~~psutil (optional): `conda install -c anaconda psutil`~~
* ~~[gradio](https://www.gradio.app) (optional): `pip install gradio`~~

### How to use:
1. **Configure setting.py**
    * properties of captcha image
    * num of image you want to generate
    * where to save imgs

2. Generate images. 
>Note: **PLEASE ENSURE YOU HAVE ENOUGH DISK SPACE**.
```
$ python generate_img.py
```
3. (optional) Train the model.
```
$ python train.py
```
4. Validate
```
$ python valid.py
```

### TODO:
- [ ] Add train accuracy during training
- [ ] provide online version

