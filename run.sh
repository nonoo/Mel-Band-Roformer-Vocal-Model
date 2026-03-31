#!/bin/bash
self=`readlink "$0"`
if [ -z "$self" ]; then
	self=$0
fi
scriptname=`basename "$self"`
scriptdir=${self%$scriptname}

cd $scriptdir

.venv/bin/python inference.py \
	--config_path configs/config_vocals_mel_band_roformer.yaml \
	--model_path MelBandRoformer.ckpt "$@"
