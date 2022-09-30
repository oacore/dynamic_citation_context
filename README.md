# dynamic_citation_context
This repository contains datasets and source code for the AACL-IJCNLP, 2022 paper "Dynamic Citation Extraction for Citation Classification"

## Requirements 
```
transformers==4.5.1
```

## Citation Class labels

```
BACKGROUND - 0
COMPARES_CONTRASTS - 1
EXTENSION - 2
FUTURE - 3
MOTIVATION - 4
USES - 5
```
## Extract Fixed Context

```
python src/dynamic_context/fixed_context_extraction.py --data_set {sdp_act/acl_arc} --train_size {3000/1647}
```


## Extract Dynamic Context

```
python src/dynamic_context/dynamic_context_extraction.py --data_set {sdp_act/acl_arc} --train_size {3000/1647} --context_type {non_contiguous/contiguous} --exp_type {exp1/2/3/4}
```
exp's represents the different combinations of the citing/cited embeddings, determined using SPECTER/SciNCL.

exp1 - ```(cited title, cited abstract) ; (citing title, sent_i)```

exp2 - ```(cited title); (sent_i)```

exp3 - ```(cited title, cited abstarct) ; (sent_i)```

exp4 - ```(cited title, cited abstract) ; (cited title, sent_i)```

The path to the models (SPECTER and SciNCL) are defined in 

```
src/dynamic_context/__init__.py
```

## Citation Classification

```
python src/citation_classification.py \
--data_dir data/non_contiguous_sdp_act \
--output_dir output/context_prev \
--context_window context_prev \
--task_name citation_function \
--model_name_or_path allenai/scibert_scivocab_uncased \
--do_train \
--num_train_epochs 5
```
--data_dir - path to the generated dynamic or fixed context.

--context_window - defined by ```processors_fixed_context and processors_dynamic_context under src/classification_utils.py``` 
## License

```
SDP-ACT is derived from CORE, the contents of which is provided under the ODC Attribution License (ODC-BY 1.0).
```
