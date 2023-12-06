import os

from detectron2.data import MetadataCatalog

from .graspnet import register_graspnet_instances
from .graspnet_meta import get_graspnet_instances_meta
from .voc_coco import register_voc_coco

_GRASPNET_OS_SPLITS = {
    "graspnet_train": ("graspnet_os/images", "graspnet_os/annotations/graspnet_os_train.json"),
    "graspnet_test_1": ("graspnet_os/images", "graspnet_os/annotations/graspnet_os_test_1.json"),
    "graspnet_test_2": ("graspnet_os/images", "graspnet_os/annotations/graspnet_os_test_2.json"),
    "graspnet_test_3": ("graspnet_os/images", "graspnet_os/annotations/graspnet_os_test_3.json"),
    "graspnet_test_4": ("graspnet_os/images", "graspnet_os/annotations/graspnet_os_test_4.json"),
    "graspnet_test_5": ("graspnet_os/images", "graspnet_os/annotations/graspnet_os_test_5.json"),
    "graspnet_test_6": ("graspnet_os/images", "graspnet_os/annotations/graspnet_os_test_6.json"),
}

def register_graspnet_os(root):
    for key, (image_root, json_file) in _GRASPNET_OS_SPLITS.items():
        register_graspnet_instances(
            key,
            get_graspnet_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_opendet_voc_coco(root):
    SPLITS = [
        # VOC_COCO_openset
        ("voc_coco_20_40_test", "voc_coco", "voc_coco_20_40_test"),
        ("voc_coco_20_60_test", "voc_coco", "voc_coco_20_60_test"),
        ("voc_coco_20_80_test", "voc_coco", "voc_coco_20_80_test"),

        ("voc_coco_2500_test", "voc_coco", "voc_coco_2500_test"),
        ("voc_coco_5000_test", "voc_coco", "voc_coco_5000_test"),
        ("voc_coco_10000_test", "voc_coco", "voc_coco_10000_test"),
        ("voc_coco_20000_test", "voc_coco", "voc_coco_20000_test"),

        ("voc_coco_val", "voc_coco", "voc_coco_val"),

    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_voc_coco(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

if __name__.endswith(".custom"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
    register_graspnet_os(_root)
    register_opendet_voc_coco(_root)