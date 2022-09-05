# Labelbox Scripts
## Setup
1. Create a Python3 virtual environment here
2. Activate virtual environment
3. Install dependencies with `requirements.txt` using:
```sh
pip install -r /Labelbox/scripts/requirements.txt
```
Note: you might need to install scikit-image:
```sh
pip install scikit-image
````
You may need this step as well:
```sh
pip install -e git+https://github.com/Labelbox/pascal-voc-writer.git@master
```

There is a `test.py`function which has three main functions:
## Input:
`--name`: name of the function that can be chosen from: `sh`,`c`,`ctt`. By default, it is set to convert labelbox output to coco format 

`--file_name`: name of the output of labelbox with json format


`--image`: if you choose the `c` function and set `--image` as true, it will create a folder with the name of json file and copy the images there. This value is false by default. If you choose `ctt` function, it will create two folders for train and test images and save the images there.

`--label`: you can create a txt file with the name of labels and its ID. tomato_label.txt is an example of this file. If you don't create it, the code will automatically create it with the order of objects.

`--training_percentage`: If you choose `--name` as `ctt`, you can choose how many percentage of image you want to be in training. For example, if you use 0.9 it means 90% of images will be in training dataset and 10% in testing dataset.

## Output:
`--name=sh`:

        On terminal, the input of the json file would be shown

`--name=c`: 

        A coco json format file with the name of input +_coco.json

        If you set --image parameter as true and the 'c' function is selected, a folder with the json file name+_image

`--name=ctt`: 

        Two coco json format files with the name of input +_coco_test.json and input +_coco_trainn.json 
        
        If you set --image parameter as true and the 'ctt' function is selected, two folders with their json file for training and testing with the names of name+_image_train and name+_image_test will be created 



## Testing:

### 1. Show Current JSON file
```sh
python3 test.py --name=sh --file_name=tomato.json
```

### 2. Convert from Labelbox output to COCO JSON file
```sh
python3 test.py --name=c --file_name=tomato.json --image=True 
```

### 3. Convert from Labelbox output to COCO JSON file and creating test and train COCO JSON files and their image folders with the proportion 90 to 10
```sh
python3 test.py --name=ctt --file_name=tomato.json --image=True --training_percentage=0.9
```


