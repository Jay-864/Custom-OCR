# Custom-OCR
A deep learning-based Optical Character Recognition (OCR) system built from scratch using PyTorch. This project is designed to recognize and decode structured text from grayscale images — specifically focused on license plate recognition — using a custom-trained CRNN (Convolutional Recurrent Neural Network) architecture.
<br>
**How to use?**<br><br>
Run the <u>test.py</u> in the same directory of the extracted dataset. After successful completion, a ocr_model.pth will be created.<br>
After this run
```
python test.py --test /path/to/image
```
This will use the saved ocr_model.pth to test the image given as an input to the model<br>
The <u>accuracy.py</u> is just a script used to test the model accuracy.<br><br>
Dataset link: [Click Here!](https://www.kaggle.com/datasets/nickyazdani/license-plate-text-recognition-dataset)
