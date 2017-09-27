#!/bin/sh

GCP_REGION="asia-east1"

PROJECT_ID=$(gcloud config list project --format "value(core.project)")

BUCKET_NAME="image_style_transfer"
# BUCKET_NAME=${PROJECT_ID}-ml
TRAIN_BUCKET=gs://$BUCKET_NAME

# バケットを作成する
gsutil mb -l $GCP_REGION $TRAIN_BUCKET

# 画像データをアップロードする
gsutil cp -r image $TRAIN_BUCKET

JOB_NAME=image_style_transfer_$(date +%Y%m%d%H%M%S)
GCP_SOURCE_DATE_DIR=$TRAIN_BUCKET/image
CONTENT_DATA=$GCP_SOURCE_DATE_DIR/input/input_01.jpg
STYLE_DATA=$GCP_SOURCE_DATE_DIR/style/style_01.jpg
gcloud ml-engine jobs submit training $JOB_NAME \
    --package-path=trainer \
    --module-name=trainer.task \
    --staging-bucket="$TRAIN_BUCKET" \
    --region=$GCP_REGION \
    -- \
    --train-steps 1000 \
    --content-file $CONTENT_DATA \
    --style-file $STYLE_DATA \
    --output-file $TRAIN_BUCKET/output/synthesized_image.png

