#!/usr/bin/env python3

# import labelbox2coco library
#import labelbox2coco as lb2co
from labelbox2coco import COCO_format
import os
import sys
##################################test contents##################3
import json
import argparse
#  labelbox_output = the file path of the Labelbox JSON export
# coco_output = the output

def read_classes(label_folder):
    with open(label_folder) as f:
        lines = f.readlines()
    label_dict ={}
    for line in lines:
        name = line[:line.rfind("=")]
        number =line[line.rfind("=")+1:line.rfind("\n")]
        label_dict[name] = number
    print(label_dict)
    return label_dict


def show_coco_format(coco_output):
    print(coco_output)
    with open(coco_output, 'r') as j:
        contents = json.loads(j.read())
        print(contents)

def convert_labelbox_coco(labelbox_output, image_record, label_dict):


    coco_format = COCO_format.COCO_format(labelbox_output = labelbox_output, label_dict = label_dict, image_record = image_record)

    coco_format.read_labelbox_format()

    #(self, labelbox_output = "", label_dict = "", training_percentage = 1, test_flag = False, image_record = False)



def convert_labelbox_coco_train_test(labelbox_output, image_record, label_dict, training_percentage):

    coco_format_train = COCO_format.COCO_format(labelbox_output = labelbox_output, label_dict = label_dict, training_percentage = training_percentage,
    image_record = image_record)
    coco_format_train.read_labelbox_format()
    

    coco_format_test = COCO_format.COCO_format(labelbox_output = labelbox_output, label_dict = label_dict, training_percentage = training_percentage, 
    test_flag = True, image_record = image_record)
    coco_format_test.read_labelbox_format()
    
    #lb2co.from_json_train_test(image_data_train, image_data_test, labelbox_output, coco_output_train,coco_output_test, training_percentage)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help ='the function name', default="c",choices=["c", 
    "ctt", "sh"])
    parser.add_argument('--file_name', type=str, help = 'the labelbox output file path',default="tomato.json")
    parser.add_argument('--label', type=str, help ='A dictionary for labelled data', default="tomato_label.txt")
    parser.add_argument('--image', type=str, help ='The images of labelled data would be saved in a folder with the name <labelbox_json_file+_image>', 
    default="False",choices=["True","False"])
    parser.add_argument('--training_percentage', type=float, help ='Percentage of training', default=0.8)
    args = parser.parse_args()
    
    label_dict = {}
    if((args.name == "c" or args.name == "ctt") and os.path.exists(args.label)):
        label_dict = read_classes(args.label)

    if(args.name == "c"):
        convert_labelbox_coco(args.file_name,args.image, label_dict)

    elif(args.name == "ctt"):
        convert_labelbox_coco_train_test(args.file_name, args.image, label_dict, args.training_percentage)
    
    elif(args.name == "sh"):
        show_coco_format(args.file_name) #pass file path 

if __name__ == '__main__':
    main()
