import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import os
from matplotlib import pyplot as plt

coco = COCO('./annotations/instances_default.json')
img_dir = './dataset/'

for id in coco.imgs:
    one = coco.imgs[id]
    try:
        image = np.array(Image.open(os.path.join(img_dir + 'furcullaria/', one['file_name'])))
    except Exception:
        pass
    try:
        image = np.array(Image.open(os.path.join(img_dir + 'zostera/', one['file_name'])))
    except Exception:
        pass

    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds=one['id'], catIds=cat_ids, iscrowd=None)

    anns = coco.loadAnns(anns_ids)

    mask = coco.annToMask(anns[0])
    for i in range(len(anns)):
        mask += coco.annToMask(anns[i])

    print(mask)
    plt.imshow(mask)
    plt.show()

