# Demos4CV
Demos for Computer Vision.
[geg](./captcha_image_recognition/README.md)

# Features:
* support number, upper case and lower case character
* don't need to preprocess captcha imgs

# To-do:
- [ ] resize input img so that this model can accept any size of img
- [ ] code format and explaination
- [ ] add test.py
- [ ] save generate img data to memory directly

# How to use
1. Make sure you have installed these python modules
    * NumPy
    * PyTorch
    * captcha
    * you may need to install PIL
2. configure setting.py
    * How many character nums you want to have
    * Which character you want to include
    * how many imgs to generate for train, valid, test
3. Generate imgs
    ```
    python generate_img.py
    ```
4. Train the model
    ```
    python train.py
    ```
5. Valid the model
    ```
    python valid.py
    ```

# Reference: 




