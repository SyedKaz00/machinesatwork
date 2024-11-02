import os
import json
import uuid
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import requests
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from shapely.geometry import MultiPolygon
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.coco import load_coco_json
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer

from labelbox import Client, LabelingFrontend, OntologyBuilder
from labelbox.data.serialization import COCOConverter, NDJsonConverter
from labelbox.schema.model import Model
from labelbox.data.metrics.group import get_label_pairs
from labelbox.data.annotation_types import (
    Mask,
    MaskData,
    ObjectAnnotation,
    LabelList,
    Point,
    Rectangle,
    Polygon,
    ImageData,
    Label,
    ScalarMetric
)
from labelbox.data.metrics import (
    feature_miou_metric,
    feature_confusion_matrix_metric
)

with open('./coco_utils.py', 'w' ) as file:
    helper = requests.get("https://raw.githubusercontent.com/Labelbox/labelbox-python/develop/examples/integrations/detectron2/coco_utils.py").text
    file.write(helper)
from coco_utils import visualize_coco_examples, visualize_object_inferences, partition_coco


API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbDJyY2Z4aXQwNTV6MDdhNTZlYm9hY2J1Iiwib3JnYW5pemF0aW9uSWQiOiJjbDJuNDJuZzBsbDY5MHpib2h1NWczdXN5IiwiYXBpS2V5SWQiOiJjbDV2MmNjZmI1NGRnMDg1NjAyeGFmOWQ4Iiwic2VjcmV0IjoiZTVhOTQ0YWEzMTI3NTVhZjk4OTJhOGQyZDlhMmRlYWQiLCJpYXQiOjE2NTg0MDk3NDQsImV4cCI6MjI4OTU2MTc0NH0.jFzHgh6CfeV1dhZTxQvRA2o6aAW8h79Jym2CGTOvN9A"
# For training:
project_id = "cl2n46a7qloaa0z9l7r2nfzgv"
# The model will make predictions on the following dataset 
# and upload predictions to a new project for model assisted labeling.
mal_dataset_id = "cl2n46a87load0z9l4yjhdub6"

print("Hello wrold")

# Based on:
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=Ya5nEuMELeq8


with open('./coco_utils.py', 'w') as file:
    helper = requests.get(
        "https://raw.githubusercontent.com/Labelbox/labelbox-python/develop/examples/integrations/detectron2/coco_utils.py").text
    file.write(helper)


client = Client(api_key=API_KEY)

# These don't need to be set/ Set these up every time you get new data from labelbox
image_root = "/home/kieran/PycharmProjects/TheMachinesDoBeWorking/image_root"
mask_root = "/home/kieran/PycharmProjects/TheMachinesDoBeWorking/mask_root"
train_json_path = '/home/kieran/PycharmProjects/TheMachinesDoBeWorking/exportCorrect_train_coco.json'
test_json_path = '/home/kieran/PycharmProjects/TheMachinesDoBeWorking/exportCorrect_test_coco.json'
train_test_split = [0.8, 0.2]
model_zoo_config = "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"

# These can be set to anything. As long as this process doesn't have
# another dataset with these names
train_ds_name = "custom_coco_train1"
test_ds_name = "custom_coco_test1"

model_name = "Sat_detectron_object_model"

proj = client.get_project(project_id)
for path in [image_root, mask_root]:
    if not os.path.exists(path):
        os.mkdir(path)
        # Could bug out just manually create the directories

# labels = proj.label_generator()
# val_labels = [next(labels) for idx in range(25)]


# coco = COCOConverter.serialize_instances(    labels = labels,     image_root = image_root )
# Try this with the coco json file, dont do anything with the alreadyd downlaoded images, it might fuck up
# Set location to where the converter has files
with open(
        '/home/kieran/vsWorkspace/projectsAtWork/machines-at-work-satellite-imagery/Labelbox-master/scripts/50922_coco.json') as f:
    data = json.load(f)
coco = data
# train_partition, test_partition = partition_coco(coco, splits=[0.8, 0.2])
#
# for parition, file_name in [[train_partition, train_json_path], [test_partition, test_json_path]]:
#     with open(file_name, 'w') as file:
#         json.dump(parition['instance'], file)

register_coco_instances(train_ds_name, {} , train_json_path, "/home/kieran/PycharmProjects/TheMachinesDoBeWorking/exportCorrect_image_train")
register_coco_instances(test_ds_name , {} , test_json_path, "/home/kieran/PycharmProjects/TheMachinesDoBeWorking/exportCorrect_train_train")


MetadataCatalog.get(test_ds_name).thing_classes = {r['id'] : r['name'] for r in coco['categories']}
test_json = load_coco_json(test_json_path, image_root)
#visualize_coco_examples(MetadataCatalog.get(test_ds_name), test_json , resize_dims = (768, 512), max_images = 2)

# help(model_zoo)

# Clear metadata so detectron recomputes.
if hasattr(MetadataCatalog.get(train_ds_name), 'thing_classes'):
    del MetadataCatalog.get(train_ds_name).thing_classes
if hasattr(MetadataCatalog.get(test_ds_name), 'thing_classes'):
    del MetadataCatalog.get(test_ds_name).thing_classes

# Set model config.
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_zoo_config))
cfg.DATASETS.TRAIN = (train_ds_name,)
cfg.DATASETS.TEST = ()
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_zoo_config)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 120
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(coco['categories'])

print("Cfg output below")
print(cfg)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)

trainer.resume_or_load(resume=False)
trainer.train()

# Use this for validation if you would like..
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator(test_ds_name)
val_loader = build_detection_test_loader(cfg, test_ds_name)
print(inference_on_dataset(trainer.model, val_loader, evaluator))

# We can use `Visualizer` to draw the predictions on the image.

# Export Data From Catalog or another dataset here..
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)
test_json = load_coco_json(test_json_path, image_root)
del MetadataCatalog.get(test_ds_name).thing_classes
MetadataCatalog.get(test_ds_name).thing_classes = {idx : r['name'] for idx, r in enumerate(coco['categories'])}


visualize_object_inferences(MetadataCatalog.get(test_ds_name),test_json, predictor)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
predictor = DefaultPredictor(cfg)
# for single test image


def get_label(image):
  # This is a bit slow, there is some i/o for downloading the images but then inference is slow
  # Runs inference on an image and returns a label
    res = predictor(image.value)
    annotations = []
    for idx in range(len(res['instances'])):
        mask = MaskData.from_2D_arr(res['instances'][idx].pred_masks[0].cpu().numpy().astype(np.uint8))
        mask = Mask(mask = mask, color = (1,1,1))
        geom = mask.shapely.buffer(0).simplify(3)
        if isinstance(geom, MultiPolygon):
            geom = geom.convex_hull

        annotations.append(ObjectAnnotation(
            name = MetadataCatalog.get(test_ds_name).thing_classes[res['instances'][idx].pred_classes[0].cpu().numpy().item()],
            value = Polygon(points = [Point(x=x,y=y) for x,y in list(geom.exterior.coords)]),
        ))
    return Label(data = image, annotations = annotations)

# Allows us to upload local images to labelbox
signer = lambda _bytes: client.upload_data(content=_bytes, sign=True)


labels_mea = LabelList()

with ThreadPoolExecutor(4) as executor:
    futures = [executor.submit(get_label, label.data) for label in val_labels]
    for future in tqdm(as_completed(futures)):
        labels_mea.append(future.result())

labels_mea.add_url_to_masks(signer) \
      .add_url_to_data(signer) \
      .assign_feature_schema_ids(OntologyBuilder.from_project(proj))


# If the model already exists fetch it with the following:

model = next(client.get_models(where = Model.name == model_name), None)
if model is None:
    model = client.create_model(model_name, ontology_id=proj.ontology().uid)


# Increment model run version if it exists. Otherwise use the initial 0.0.0
model_run_names = [model_run.name for model_run in model.model_runs()]
if len(model_run_names):
    model_run_names.sort(key=lambda s: [int(u) for u in s.split('.')])
    latest_model_run_name = model_run_names[-1]
    model_run_suffix = int(latest_model_run_name.split('.')[-1]) + 1
    model_run_name = ".".join([*latest_model_run_name.split('.')[:-1], str(model_run_suffix)])
else:
    model_run_name = "0.0.0"

print(f"Model Name: {model.name} | Model Run Version : {model_run_name}")
model_run = model.create_model_run(model_run_name)
model_run.upsert_labels([label.uid for label in val_labels])

pairs = get_label_pairs(val_labels, labels_mea, filter_mismatch=True)
for (ground_truth, prediction) in pairs.values():
    metrics = []
    metrics.extend(feature_miou_metric(ground_truth.annotations, prediction.annotations))
    metrics.extend(feature_confusion_matrix_metric(ground_truth.annotations, prediction.annotations))
    prediction.annotations.extend(metrics)


upload_task = model_run.add_predictions(f'diagnostics-import-{uuid.uuid4()}', NDJsonConverter.serialize(labels_mea))
upload_task.wait_until_done()
print(upload_task.state)
print(upload_task.errors)


for idx, model_run_data_row in enumerate(model_run.model_run_data_rows()):
    if idx == 5:
        break
    print(model_run_data_row.url)

# Some additional unlabeled data rows
dataset = client.get_dataset(mal_dataset_id)


# Use ThreadPoolExecutor to parallelize image downloads.
# This is still a bit slow due to the amount of processing for each data row.
# For larger datasets this has to leverage multiprocessing.

labels_mal = LabelList()
with ThreadPoolExecutor(4) as executor:
    data_rows = dataset.data_rows()
    images = [ImageData(url = data_row.row_data, uid = data_row.uid, external_id = data_row.external_id) for data_row in data_rows]
    futures = [executor.submit(get_label, image) for image in images]
    for future in tqdm(as_completed(futures)):
        labels_mal.append(future.result())


# Create a new project for upload
project = client.create_project(name = "detectron_mal_project")
editor = next(
    client.get_labeling_frontends(where=LabelingFrontend.name == 'editor'))
project.setup(editor, labels_mal.get_ontology().asdict())
project.enable_model_assisted_labeling()
project.datasets.connect(dataset)


labels_mal.add_url_to_masks(signer) \
      .add_url_to_data(signer) \
      .assign_feature_schema_ids(OntologyBuilder.from_project(project))

ndjsons = list(NDJsonConverter.serialize(labels_mal))
upload_task = project.upload_annotations(name=f"upload-job-{uuid.uuid4()}",
                                         annotations=ndjsons,
                                         validate=False)
# Wait for upload to finish
upload_task.wait_until_done()
# Review the upload status
print(upload_task.errors)
