import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split
from data import DATA_PROCESSED_DIR
from dynamic_context_contiguous import processors_dynamic_context_contiguous
from dynamic_context_non_contiguous import processors_dynamic_context_non_contiguous


def extract_dynamic_context(data_df, context_type, exp_type):

    dynamic_contexts_prev = list()
    dynamic_contexts_next = list()
    dynamic_contexts_combined = list()
    data_df['cite_context_paragraph'] = data_df['cite_context_paragraph'].apply(eval)
    for idx, row in data_df.iterrows():

        paragraph = row['cite_context_paragraph']
        citation_context = row['citation_context']
        citing_title = row['citing_title']
        cited_title = row['cited_title']

        if type(row['cited_abstract']) is not str:
            cited_abstract = ''

            print('abstract None')
        else:
            cited_abstract = row['cited_abstract']

        if context_type == 'non_contiguous':
            processors = processors_dynamic_context_non_contiguous[exp_type](citing_title, cited_title,
                                                                             citation_context,
                                                                             cited_abstract, paragraph)
            cited_abstract_emb, similarity_citation_context = processors.extract_embeddings()
            dynamic_context_prev, dynamic_context_next, dynamic_context_combined = \
                processors.extract_context(cited_abstract_emb, similarity_citation_context)

            dynamic_contexts_prev.append(dynamic_context_prev)
            dynamic_contexts_next.append(dynamic_context_next)
            dynamic_contexts_combined.append(dynamic_context_combined)

        else:
            processors = processors_dynamic_context_contiguous[exp_type](citing_title, cited_title,
                                                                         citation_context,
                                                                         cited_abstract, paragraph)
            cited_abstract_emb, similarity_citation_context = processors.extract_embeddings()
            dynamic_context_prev, dynamic_context_next, dynamic_context_combined = \
                processors.extract_context(cited_abstract_emb, similarity_citation_context)

            print(dynamic_context_prev)
            print(dynamic_context_next)
            print(dynamic_context_combined)
            dynamic_contexts_prev.append(dynamic_context_prev)
            dynamic_contexts_next.append(dynamic_context_next)
            dynamic_contexts_combined.append(dynamic_context_combined)

    return dynamic_contexts_prev, dynamic_contexts_next, dynamic_contexts_combined


def main():
    parser = argparse.ArgumentParser(description="Code to generate dynamic context from paragraph")
    ## Required parameters
    parser.add_argument("--data_set", default='sdp_act', type=str, required=True,
                        help="Dataset used for the experiment - sdp_act or acl_arc")
    parser.add_argument("--train_size", default=3000, type=int,
                        help="Length of trainset. sdp_act = 3000 and acl_arc = 1647")
    #parser.add_argument("--model_path", default='allenai/specter', type=str, required=True,
    #                    help="Path to pre-trained model, ex: allenai/specter")
    parser.add_argument("--context_type", default='non_contiguous', type=str, required=True,
                        help="type of the context used - non_contiguous or contiguous")
    parser.add_argument("--exp_type", default='exp1', type=str, required=True,
                        help="exp1 - (cited title, abstract ; citing title, sent), exp2 - (cited title; sent), "
                             "exp3 - (cited title, cited abstract ; sent), exp4 - (cited title, abstract ; cited "
                             "title, sent)")

    args = parser.parse_args()
    DATASET_DIR = DATA_PROCESSED_DIR / f"{args.data_set}"
    if not os.path.exists(os.path.join(DATA_PROCESSED_DIR, f"{args.context_type}_{args.data_set}_{args.exp_type}")):
        os.makedirs(os.path.join(DATA_PROCESSED_DIR, f"{args.context_type}_{args.data_set}_{args.exp_type}"))

    OUTPUT_DIR = os.path.join(DATA_PROCESSED_DIR, f"{args.context_type}_{args.data_set}_{args.exp_type}")

    for dataset in ["train", "test"]:
        data_df = pd.read_csv(DATASET_DIR / f"{dataset}.txt", sep="\t", engine="python", dtype=object)
        dynamic_contexts_prev, dynamic_contexts_next, dynamic_contexts_combined = \
            extract_dynamic_context(data_df, args.context_type, args.exp_type)
        data_df['dynamic_context_prev'] = dynamic_contexts_prev
        data_df['dynamic_context_next'] = dynamic_contexts_next
        data_df['dynamic_contexts_combined'] = dynamic_contexts_combined

        if dataset == 'train':
            X_train, X_valid, _, _ = train_test_split(data_df, [0] * args.train_size, test_size=0.166666666,
                                                      random_state=30)
            X_train.to_csv(OUTPUT_DIR + '/train.txt', sep='\t', encoding='utf-8', index=False)
            X_valid.to_csv(OUTPUT_DIR + '/valid.txt', sep='\t', encoding='utf-8', index=False)

        else:
            data_df.to_csv(OUTPUT_DIR + f"/{dataset}.txt", sep='\t', encoding='utf-8',
                             index=False)


if __name__ == '__main__':
    main()
