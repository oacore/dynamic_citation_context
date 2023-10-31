# dynamic_citation_context
This repository contains datasets and source code for the AACL-IJCNLP, 2022 paper [Dynamic Context Extraction for Citation Classification](https://aclanthology.org/2022.aacl-main.41.pdf).

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

## Citation
```
@inproceedings{nambanoor-kunnath-etal-2022-dynamic,
    title = "Dynamic Context Extraction for Citation Classification",
    author = "Nambanoor Kunnath, Suchetha  and
      Pride, David  and
      Knoth, Petr",
    editor = "He, Yulan  and
      Ji, Heng  and
      Li, Sujian  and
      Liu, Yang  and
      Chang, Chua-Hui",
    booktitle = "Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = nov,
    year = "2022",
    address = "Online only",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.aacl-main.41",
    pages = "539--549",
    abstract = "We investigate the effect of varying citation context window sizes on model performance in citation intent classification. Prior studies have been limited to the application of fixed-size contiguous citation contexts or the use of manually curated citation contexts. We introduce a new automated unsupervised approach for the selection of a dynamic-size and potentially non-contiguous citation context, which utilises the transformer-based document representations and embedding similarities. Our experiments show that the addition of non-contiguous citing sentences improves performance beyond previous results. Evalu- ating on the (1) domain-specific (ACL-ARC) and (2) the multi-disciplinary (SDP-ACT) dataset demonstrates that the inclusion of additional context beyond the citing sentence significantly improves the citation classifi- cation model{'}s performance, irrespective of the dataset{'}s domain. We release the datasets and the source code used for the experiments at: \url{https://github.com/oacore/dynamic_citation_context}",
}
```

