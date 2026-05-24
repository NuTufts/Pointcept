#!/bin/bash

CONTAINER=/projects/u6jo/containers/pointcept-sandbox/
SQASHFILE=/projects/u6jo/datasets/combined_pretrain-sonata-v7-extbnb-larmatch.sqsh

apptainer shell --nv --bind $SQASHFILE:/data:image-src=,ro  $CONTAINER