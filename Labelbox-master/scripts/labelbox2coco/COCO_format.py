
#Structure of COCO format dataset:
#https://towardsdatascience.com/how-to-work-with-object-detection-datasets-in-coco-format-9bf4fb5848a4
import json
import datetime as dt
import logging
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
import requests
from PIL import Image, ImageOps
from skimage import measure  
import numpy as np
import os
import cv2
from termcolor import colored

class COCO_format():
    def __init__(self, labelbox_output = "", label_dict = {}, training_percentage = 1, test_flag = False, image_record = False):
        self.labelbox_output_path = labelbox_output
        self._label_data = []
        self._coco = []
        self.label_dict = label_dict
        self.training_percentage = training_percentage
        self.test_flag = test_flag
        self.start_data = 0 #Make it flexible for training and testing
        self.end_data = 0
        self.image_record = image_record
        self.prefix = ""

        self._read_jason_file()
        self._set_coco()
        if(len(self.label_dict)): ##If the user send a dictionary of label classes
            self.set_category()
        self.count_labelled_img() ##Some images does not have any labelled data, this part will remove those images


    def _read_jason_file(self):
        with open(self.labelbox_output_path, 'r') as f:
            self._label_data = json.loads(f.read())

    def _set_coco(self):
        self._coco = {
        'info': None,
        'licenses': [],
        'categories': [],
        'images': [],
        'annotations': []
        }   

        self._coco['licenses'] = {
            'id': 1,
            'name': "University of Auckland",
            'url': 'labelbox.com'
        }
        self._coco['info'] = {
            'year': dt.datetime.now(dt.timezone.utc).year,
            'version': None,
            'description': self._label_data[0]['Project Name'],
            'contributor': self._label_data[0]['Created By'],
            'url': 'labelbox.com',
            'date_created': dt.datetime.now(dt.timezone.utc).isoformat()
        }

    def set_category(self):
        for class_name in self.label_dict: 
            category = {
                'supercategory': "type",
                'id': self.label_dict[class_name],
                'name': class_name
            }
            self._coco['categories'].append(category)
    
    def save_image(self, image_name, image):
    
        image_path = os.path.join(os.path.dirname(os.path.abspath(self.labelbox_output_path)), 
        os.path.basename(self.labelbox_output_path)[:-5] + "_image" + self.prefix)
        if(os.path.exists(image_path)==False):
            os.mkdir(image_path)
            print(colored(("Image directory: "+ image_path ), 'green'))

        
        image.save(os.path.join(image_path, image_name ))

        print(colored(("Image "+ image_name + " is saved" ), 'magenta'))
    
    def count_labelled_img(self):
        count_img = 0
        temp_label_data = []
        for data in self._label_data:
            if (len(data['Label'].keys()) != 0):
                temp_label_data.append(data)
                count_img = count_img + 1        
        self._label_data = temp_label_data
        if (count_img == 0):
            print("No image labelled")
            return

        print(colored("Number of images with labels: %d" % (len(self._label_data)), 'green'))
        if(self.training_percentage < 1 and self.training_percentage >=0):##if the coco dataset will be used for testing
            if (self.test_flag): 
                self.start_data = int(count_img*self.training_percentage)
                self.end_data = len(self._label_data)
                self.prefix = "_test"
            else: ##if the coco dataset will be used for training
                self.start_data = 0
                self.end_data = int(float(count_img)*self.training_percentage)
                self.prefix = "_train"
        elif(self.training_percentage == 1): ##Default creating coco data set
                self.start_data = 0
                self.end_data = len(self._label_data)
                self.prefix = ""
        else:
            print(colored("*********************************Wrong percentage*****************************", 'red'))
 
    def write_coco_format(self): #Save the dataset in CoCo format
        coco_output = os.path.join(os.path.dirname(os.path.abspath(self.labelbox_output_path)), 
        os.path.basename(self.labelbox_output_path)[:-5] + self.prefix +"_coco.json")
        with open(coco_output, 'w+') as f:
            f.write(json.dumps(self._coco, indent=4, separators=(',', ': ')))   
        f.close()     
        

 #In the case that user does not enter the dictionary of labelled class, 
 #The dictionary will be created based on the order of labells that will be appered while reading the labelbox output
    def set_default_category(self,url):        
        #try:
        # check if label category exists in 'categories' field
        #    cat_id = [c['id'] for c in self._coco['categories']
        #        if c['supercategory'] == url['title']][0]
        #except IndexError as e:
        cat_id = len(self._coco['categories']) + 1
        category = {
            'supercategory': url['title'],
            'id': len(self._coco['categories']) + 1,
            'name': url['title']
        }
        self.label_dict[url['title']] = len(self._coco['categories']) + 1
        self._coco['categories'].append(category)
        
        
    def read_labelbox_format(self):
        for counter_data in range(self.start_data,self.end_data):
            data = 	self._label_data[counter_data]
##################################Image information#######################################            
        # Download and get image name
            try:
                response = requests.get(data['Labeled Data'], stream=True)
            except requests.exceptions.MissingSchema as e:
                logging.exception(('"Labeled Data" field must be a URL. '
                            'Support for local files coming soon'))
                continue
            except requests.exceptions.ConnectionError as e:
                logging.exception('Failed to fetch image from {}'
                            .format(data['Labeled Data']))
                continue
            #info about image
            response.raw.decode_content = True
            im = Image.open(response.raw)
            width, height = im.size

            #save the images in image_path folder
            first_part = data['Labeled Data'][:data['Labeled Data'].find("?")]
            image_name = first_part[first_part.rfind("-")+1:]

            if(self.image_record):
                self.save_image(data['Dataset Name'] + "_" + image_name , im)

            image = {
                "id": data['ID'],
                "width": width,
                "height": height,
                "file_name": data['Dataset Name'] + "_" +image_name,#data['Labeled Data'],
                "license": None,
                "flickr_url": data['Labeled Data'],
                "coco_url": data['Labeled Data'],
                "date_captured": None,
            }
            self._coco['images'].append(image)

##################################Annotation information#######################################          
            # convert WKT multipolygon to COCO Polygon format
            for cat in data['Label'].keys():
    ##############read segmtentation url and create the submask#################
            #The type of labelling is mask and the mask is a url
                for url in data['Label'][cat]:
                    if url['title'] not in self.label_dict: #If the label dictionary was not set
                        self.set_default_category(url)

            #get the URL and read the image
                    response_url = requests.get(url['instanceURI'], stream=True)
                    response_url.raw.decode_content = True
                    im_mask = Image.open(response_url.raw)
            #convert the image to binary
                    thresh = 50
                    fn = lambda x : 255 if x > thresh else 0
                    binaryMask= im_mask.convert('L').point(fn, mode='1')
                    binaryMask = ImageOps.expand(binaryMask,border=1,fill='black')

                    #convert the mask to the polygons and segmentation
                    [polygons, segmentation] = self.create_sub_mask_annotation(binaryMask)
                    #print(polygons)
            #For each polygon save the information
                    self.save_info_polygon(polygons, height, url,data)
                    
        self.write_coco_format()
        
    
    def save_info_polygon(self, polygons, height, url,data):
        for m in polygons:
            segmentation = []
            for x, y in m.exterior.coords:
                segmentation.extend([x, height-y])
            annotation = {
                "id": len(self._coco['annotations']) + 1,
                "image_id": data['ID'],
                "category_id": self.label_dict[url['title']], #url['value'], #label[url['value']], #this part can be changes
                "segmentation": [segmentation],
                "area": m.area,  # float
                "bbox": [m.bounds[0], m.bounds[1],
                    m.bounds[2]-m.bounds[0],
                    m.bounds[3]-m.bounds[1]],
            "iscrowd": 0
            }
            self._coco['annotations'].append(annotation)


    def create_sub_masks(self, mask_image, width, height):
    # Initialize a dictionary of sub-masks indexed by RGB colors
        sub_masks = {}
        for x in range(width):
            for y in range(height):
                # Get the RGB values of the pixel
                pixel = mask_image.getpixel((x,y))[:3]

                # If the pixel is not black...
                if pixel != (0, 0, 0):
                    # Check to see if we've created a sub-mask...
                    pixel_str = str(pixel)
                    sub_mask = sub_masks.get(pixel_str)
                    if sub_mask is None:
                    # Create a sub-mask (one bit per pixel) and add to the dictionary
                        # Note: we add 1 pixel of padding in each direction
                        # because the contours module doesn't handle cases
                        # where pixels bleed to the edge of the image
                        sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                    # Set the pixel value to 1 (default is 0), accounting for padding
                    sub_masks[pixel_str].putpixel((x+1, y+1), 1)

        return sub_masks

    def create_sub_mask_annotation(self, sub_mask):
        # Find contours (boundary lines) around each sub-mask
        # Note: there could be multiple contours if the object
        # is partially occluded. (E.g. an elephant behind a tree)
        contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation='low')

        polygons = []
        segmentations = []
        j = 0
        for contour in contours:
            # Flip from (row, col) representation to (x, y)
            # and subtract the padding pixel
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)

            # Make a polygon and simplify it
            poly = Polygon(contour)
            poly = poly.simplify(1.0, preserve_topology=False)
            
            if(poly.is_empty):
                # Go to next iteration, dont save empty values in list
                continue

            polygons.append(poly)

            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)
        
        return polygons, segmentations