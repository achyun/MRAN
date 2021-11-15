# Multimodal Relations Auxiliary Network (MRAN)  

## preprocess
+ Build corpus
~~~~bash
python preprocess/build_vocab.py --dataset CFO -H 2
~~~~
+ Extract appearance feature
~~~~bash
python preprocess/extract_feat.py --dataset CFO --feature_type appearance --image_height 224 --image_width 224 --gpu_id 0
~~~~
+ Extract motion feature
~~~~bash
python preprocess/extract_feat.py --dataset CFO --feature_type motion --image_height 112 --image_width 112 --gpu_id 0
~~~~
+ Extract sign language feature
~~~~bash
python preprocess/extract_feat.py --dataset CFO --feature_type hand --image_height 112 --image_width 112 --gpu_id 0
~~~~
+ Extract audio feature
~~~~bash
python preprocess/extract_feat.py --dataset CFO --feature_type audio --gpu_id 0
~~~~
