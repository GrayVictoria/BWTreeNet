# BWTreeNet
Mapping countrywide historical tree cover using semantic segmentation

## For user

### GuiTest
GuiTest is used for testing WFV images, contains all the testing strategy as illustrated in our manuscript.
You can use SwissTest.py to map other B&W images.

Besides, if you want to train the B&WTreeNet, just use the model file in /GuiTest/model/BWTreeNet.py
And if you want to get the weight, please contact us.

### Luminance Enhancer
This project is used for changing the luminance distribution for the B&W images.

We support the weight for our project.

### augmentation
This code is for image augmentation through training, which is based on the albumentation project, but we added more useful new methods.

### Other supplementary information
More segmentation results of historical B&W images:
![github_more_results1](https://github.com/user-attachments/assets/47465e1d-0c22-49cc-95b2-8e3aa559313d)
![github_more_results2](https://github.com/user-attachments/assets/ad9c34a0-23a3-44ce-9b0f-251ff9b3202e)

The input of the model is 1000pixels × 1000pixels × 1 image. The table below lists the input and output sizes corresponding to each stage of the network during feature extraction. 
![image](https://github.com/user-attachments/assets/c5ffaced-071d-4951-a1a5-94a0d607e286)


The figure below is the distribution of different study areas. 
2018 \& 2019 areas were used as training & validation set; the red areas were the manual interpretation areas based on 1980s historical which are the manual interpretation areas based on 1980s historical, these areas were marked on pixel-level by professional interpreter, and were used as testing set.
![image](https://github.com/user-attachments/assets/20d0fbdf-15a0-4b6a-ba64-480c853addc8)

The figure below is the detail distribution of mapping set, the data are from 1980 to 1985 and do not overlap and together constitute the entire
coverage of Switzerland. The total area of the mapping set is 41,008 km2 for the processing (The total area of Switzerland is 41,285 km2, and some data were missing from the swisstopo).

![image](https://github.com/user-attachments/assets/656e0260-1dcc-4cc0-8b8f-eb7937e547a3)

The table below is the detailed information on the time required for different models to predict a 1000pixels×1000pixels image, the testing time for predicting a historical B&W image with standard size 35000pixels×24000pixels image (including reading image and writing mapping result), and the time required for one training epoch is 12 mins. 

![image](https://github.com/user-attachments/assets/723bf18d-83c9-49d6-951b-172a146cc5fd)

The comparison of different image luminance(grayscale) adjustment methods:
![github_leresults](https://github.com/user-attachments/assets/7745524a-36cc-477c-9f63-fc923bbb1d3b)

Using LE to adjust the luminance of B&W image has a better effect. Compared with other methods, LE has a good enhancement effect in various scenes. In addition, B&WTreeNet also has a good effect on tree cover segmentation of other luminance-adjusted images, and the proposed LE has the best segmentation accuracy.

If you are interested in our tree mapping products throughout Switzerland 40 years ago and 80 years ago, please contact us.
