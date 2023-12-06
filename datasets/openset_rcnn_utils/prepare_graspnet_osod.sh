DATA_DIR=datasets/graspnet_os
GRASPNET_ORIGIN_DIR=datasets/graspnet

# make neccesary dirs
echo "make dirs"
mkdir -p $DATA_DIR/images

python datasets/openset_rcnn_utils/prepare_graspnet_data.py --dataset_path $GRASPNET_ORIGIN_DIR  --image_destination $DATA_DIR/images