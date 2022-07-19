import pandas as pd
from unidecode import unidecode
from collections import OrderedDict
import argparse
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split
from data import DATA_PROCESSED_DIR
import os


def get_prev_next_context(paragraph, citing_sent_index):
    context_prev = list()
    context_next = list()

    if paragraph:

        for sent in paragraph:

            non_citing_sent_index = paragraph.index(sent)
            if non_citing_sent_index < citing_sent_index:
                context_prev.append(unidecode(sent))
            elif non_citing_sent_index > citing_sent_index:
                context_next.append(unidecode(sent))
            else:
                continue

    return context_prev, context_next


def fixed_prev_next_context(prev_context, citing_sent, next_context, fixed_contexts):

    if len(prev_context) >= 3 and len(next_context) >= 3:

        extended_context = OrderedDict([

            ('cite_context_-1_sent', [prev_context[-1], citing_sent]),
            ('cite_context_sent_+1', [citing_sent]),
            ('cite_context_-1_sent_+1', [prev_context[-1], citing_sent]),
            ('cite_context_-2_sent', [prev_context[-2], prev_context[-1], citing_sent]),
            ('cite_context_sent_+2', [citing_sent]),
            ('cite_context_-2_sent_+1', [prev_context[-2], prev_context[-1], citing_sent]),
            ('cite_context_-1_sent_+2', [prev_context[-1], citing_sent]),
            ('cite_context_-3_sent', [prev_context[-3], prev_context[-2], prev_context[-1], citing_sent]),
            ('cite_context_sent_+3', [citing_sent, next_context[0], next_context[1], next_context[2]])
        ])

        fixed_contexts.append(extended_context)

    elif len(prev_context) == 2 and len(next_context) >= 3:

        extended_context = OrderedDict([

            ('cite_context_-1_sent', [prev_context[-1], citing_sent]),
            ('cite_context_sent_+1', [citing_sent]),
            ('cite_context_-1_sent_+1', [prev_context[-1], citing_sent]),
            ('cite_context_-2_sent', [prev_context[-2], prev_context[-1], citing_sent]),
            ('cite_context_sent_+2', [citing_sent]),
            ('cite_context_-2_sent_+1', [prev_context[-2], prev_context[-1], citing_sent]),
            ('cite_context_-1_sent_+2', [prev_context[-1], citing_sent]),
            ('cite_context_-3_sent', [prev_context[-2], prev_context[-1], citing_sent]),
            ('cite_context_sent_+3', [citing_sent, next_context[0], next_context[1], next_context[2]])
        ])

        fixed_contexts.append(extended_context)

    elif len(prev_context) == 1 and len(next_context) >= 3:

        extended_context = OrderedDict([

            ('cite_context_-1_sent', [prev_context[-1], citing_sent]),
            ('cite_context_sent_+1', [citing_sent, next_context[0]]),
            ('cite_context_-1_sent_+1', [prev_context[-1], citing_sent]),
            ('cite_context_-2_sent', [prev_context[-1], citing_sent]),
            ('cite_context_sent_+2', [citing_sent, next_context[0], next_context[1]]),
            ('cite_context_-2_sent_+1', [prev_context[-1], citing_sent, next_context[0]]),
            ('cite_context_-1_sent_+2', [prev_context[-1], citing_sent, next_context[0], next_context[1]]),
            ('cite_context_-3_sent', [prev_context[-1], citing_sent]),
            ('cite_context_sent_+3', [citing_sent, next_context[0], next_context[1], next_context[2]])
        ])

        fixed_contexts.append(extended_context)


    elif len(prev_context) >= 3 and len(next_context) == 2:

        extended_context = OrderedDict([

            ('cite_context_-1_sent', [prev_context[-1], citing_sent]),
            ('cite_context_sent_+1', [citing_sent, next_context[0]]),
            ('cite_context_-1_sent_+1', [prev_context[-1], citing_sent, next_context[0]]),
            ('cite_context_-2_sent', [prev_context[-2], prev_context[-1], citing_sent]),
            ('cite_context_sent_+2', [citing_sent, next_context[0], next_context[1]]),
            ('cite_context_-2_sent_+1', [prev_context[-2], prev_context[-1], citing_sent, next_context[0]]),
            ('cite_context_-1_sent_+2', [prev_context[-1], citing_sent, next_context[0], next_context[1]]),
            ('cite_context_-3_sent', [prev_context[-3], prev_context[-2], prev_context[-1], citing_sent]),
            ('cite_context_sent_+3', [citing_sent, next_context[0], next_context[1]])
        ])

        fixed_contexts.append(extended_context)

    elif len(prev_context) >= 3 and len(next_context) == 1:

        extended_context = OrderedDict([

            ('cite_context_-1_sent', [prev_context[-1], citing_sent]),
            ('cite_context_sent_+1', [citing_sent, next_context[0]]),
            ('cite_context_-1_sent_+1', [prev_context[-1], citing_sent, next_context[0]]),
            ('cite_context_-2_sent', [prev_context[-2], prev_context[-1], citing_sent]),
            ('cite_context_sent_+2', [citing_sent, next_context[0]]),
            ('cite_context_-2_sent_+1', [prev_context[-2], prev_context[-1], citing_sent, next_context[0]]),
            ('cite_context_-1_sent_+2', [prev_context[-1], citing_sent, next_context[0]]),
            ('cite_context_-3_sent', [prev_context[-3], prev_context[-2], prev_context[-1], citing_sent]),
            ('cite_context_sent_+3', [citing_sent, next_context[0]])

        ])

        fixed_contexts.append(extended_context)

    elif len(prev_context) == 2 and len(next_context) == 2:

        extended_context = OrderedDict([

            ('cite_context_-1_sent', [prev_context[-1], citing_sent]),
            ('cite_context_sent_+1', [citing_sent, next_context[0]]),
            ('cite_context_-1_sent_+1', [prev_context[-1], citing_sent, next_context[0]]),
            ('cite_context_-2_sent', [prev_context[-2], prev_context[-1], citing_sent]),
            ('cite_context_sent_+2', [citing_sent, next_context[0], next_context[1]]),
            ('cite_context_-2_sent_+1', [prev_context[-2], prev_context[-1], citing_sent, next_context[0]]),
            ('cite_context_-1_sent_+2', [prev_context[-1], citing_sent, next_context[0], next_context[1]]),
            ('cite_context_-3_sent', [prev_context[-2], prev_context[-1], citing_sent]),
            ('cite_context_sent_+3', [citing_sent, next_context[0], next_context[1]])
        ])

        fixed_contexts.append(extended_context)

    elif len(prev_context) == 1 and len(next_context) == 2:

        extended_context = OrderedDict([

            ('cite_context_-1_sent', [prev_context[-1], citing_sent]),
            ('cite_context_sent_+1', [citing_sent, next_context[0]]),
            ('cite_context_-1_sent_+1', [prev_context[-1], citing_sent, next_context[0]]),
            ('cite_context_-2_sent', [prev_context[-1], citing_sent]),
            ('cite_context_sent_+2', [citing_sent, next_context[0], next_context[1]]),
            ('cite_context_-2_sent_+1', [prev_context[-1], citing_sent, next_context[0]]),
            ('cite_context_-1_sent_+2', [prev_context[-1], citing_sent, next_context[0], next_context[1]]),
            ('cite_context_-3_sent', [prev_context[-1], citing_sent]),
            ('cite_context_sent_+3', [citing_sent, next_context[0], next_context[1]])

        ])

        fixed_contexts.append(extended_context)

    elif len(prev_context) == 2 and len(next_context) == 1:

        extended_context = OrderedDict([

            ('cite_context_-1_sent', [prev_context[-1], citing_sent]),
            ('cite_context_sent_+1', [citing_sent, next_context[0]]),
            ('cite_context_-1_sent_+1', [prev_context[-1], citing_sent, next_context[0]]),
            ('cite_context_-2_sent', [prev_context[-2], prev_context[-1], citing_sent]),
            ('cite_context_sent_+2', [citing_sent, next_context[0]]),
            ('cite_context_-2_sent_+1', [prev_context[-2], prev_context[-1], citing_sent, next_context[0]]),
            ('cite_context_-1_sent_+2', [prev_context[-1], citing_sent, next_context[0]]),
            ('cite_context_-3_sent', [prev_context[-2], prev_context[-1], citing_sent]),
            ('cite_context_sent_+3', [citing_sent, next_context[0]])
        ])

        fixed_contexts.append(extended_context)

    elif len(prev_context) == 1 and len(next_context) == 1:

        extended_context = OrderedDict([

            ('cite_context_-1_sent', [prev_context[-1], citing_sent]),
            ('cite_context_sent_+1', [citing_sent, next_context[0]]),
            ('cite_context_-1_sent_+1', [prev_context[-1], citing_sent, next_context[0]]),
            ('cite_context_-2_sent', [prev_context[-1], citing_sent]),
            ('cite_context_sent_+2', [citing_sent, next_context[0]]),
            ('cite_context_-2_sent_+1', [prev_context[-1], citing_sent, next_context[0]]),
            ('cite_context_-1_sent_+2', [prev_context[-1], citing_sent, next_context[0]]),
            ('cite_context_-3_sent', [prev_context[-1], citing_sent]),
            ('cite_context_sent_+3', [citing_sent, next_context[0]]),

        ])

        fixed_contexts.append(extended_context)

    elif len(prev_context) >= 3 and len(next_context) == 0:

        extended_context = OrderedDict([

            ('cite_context_-1_sent', [prev_context[-1], citing_sent]),
            ('cite_context_sent_+1', [citing_sent]),
            ('cite_context_-1_sent_+1', [prev_context[-1], citing_sent]),
            ('cite_context_-2_sent', [prev_context[-2], prev_context[-1], citing_sent]),
            ('cite_context_sent_+2', [citing_sent]),
            ('cite_context_-2_sent_+1', [prev_context[-2], prev_context[-1], citing_sent]),
            ('cite_context_-1_sent_+2', [prev_context[-1], citing_sent]),
            ('cite_context_-3_sent', [prev_context[-3], prev_context[-2], prev_context[-1], citing_sent]),
            ('cite_context_sent_+3', [citing_sent])
        ])

        fixed_contexts.append(extended_context)

    elif len(prev_context) == 2 and len(next_context) == 0:

        extended_context = OrderedDict([

            ('cite_context_-1_sent', [prev_context[-1], citing_sent]),
            ('cite_context_sent_+1', [citing_sent]),
            ('cite_context_-1_sent_+1', [prev_context[-1], citing_sent]),
            ('cite_context_-2_sent', [prev_context[-2], prev_context[-1], citing_sent]),
            ('cite_context_sent_+2', [citing_sent]),
            ('cite_context_-2_sent_+1', [prev_context[-2], prev_context[-1], citing_sent]),
            ('cite_context_-1_sent_+2', [prev_context[-1], citing_sent]),
            ('cite_context_-3_sent', [prev_context[-2], prev_context[-1], citing_sent]),
            ('cite_context_sent_+3', [citing_sent])
        ])

        fixed_contexts.append(extended_context)

    elif len(prev_context) == 1 and len(next_context) == 0:

        extended_context = OrderedDict([

            ('cite_context_-1_sent', [prev_context[-1], citing_sent]),
            ('cite_context_sent_+1', [citing_sent]),
            ('cite_context_-1_sent_+1', [prev_context[-1], citing_sent]),
            ('cite_context_-2_sent', [prev_context[-1], citing_sent]),
            ('cite_context_sent_+2', [citing_sent]),
            ('cite_context_-2_sent_+1', [prev_context[-1], citing_sent]),
            ('cite_context_-1_sent_+2', [prev_context[-1], citing_sent]),
            ('cite_context_-3_sent', [prev_context[-1], citing_sent]),
            ('cite_context_sent_+3', [citing_sent])
        ])

        fixed_contexts.append(extended_context)

    elif len(prev_context) == 0 and len(next_context) >= 3:

        extended_context = OrderedDict([

            ('cite_context_-1_sent', [citing_sent]),
            ('cite_context_sent_+1', [citing_sent, next_context[0]]),
            ('cite_context_-1_sent_+1', [citing_sent, next_context[0]]),
            ('cite_context_-2_sent', [citing_sent]),
            ('cite_context_sent_+2', [citing_sent, next_context[0], next_context[1]]),
            ('cite_context_-2_sent_+1', [citing_sent, next_context[0]]),
            ('cite_context_-1_sent_+2', [citing_sent, next_context[0], next_context[1]]),
            ('cite_context_-3_sent', [citing_sent]),
            ('cite_context_sent_+3', [citing_sent, next_context[0], next_context[1], next_context[2]])
        ])

        fixed_contexts.append(extended_context)

    elif len(prev_context) == 0 and len(next_context) == 2:

        extended_context = OrderedDict([

            ('cite_context_-1_sent', [citing_sent]),
            ('cite_context_sent_+1', [citing_sent, next_context[0]]),
            ('cite_context_-1_sent_+1', [citing_sent, next_context[0]]),
            ('cite_context_-2_sent', [citing_sent]),
            ('cite_context_sent_+2', [citing_sent, next_context[0], next_context[1]]),
            ('cite_context_-2_sent_+1', [citing_sent, next_context[0]]),
            ('cite_context_-1_sent_+2', [citing_sent, next_context[0], next_context[1]]),
            ('cite_context_-3_sent', [citing_sent]),
            ('cite_context_sent_+3', [citing_sent, next_context[0], next_context[1]])
        ])

        fixed_contexts.append(extended_context)

    elif len(prev_context) == 0 and len(next_context) == 1:

        extended_context = OrderedDict([

            ('cite_context_-1_sent', [citing_sent]),
            ('cite_context_sent_+1', [citing_sent, next_context[0]]),
            ('cite_context_-1_sent_+1', [citing_sent, next_context[0]]),
            ('cite_context_-2_sent', [citing_sent]),
            ('cite_context_sent_+2', [citing_sent, next_context[0]]),
            ('cite_context_-2_sent_+1', [citing_sent, next_context[0]]),
            ('cite_context_-1_sent_+2', [citing_sent, next_context[0]]),
            ('cite_context_-3_sent', [citing_sent]),
            ('cite_context_sent_+3', [citing_sent, next_context[0]])

        ])
        fixed_contexts.append(extended_context)

    elif len(prev_context) == 0 and len(next_context) == 0:

        extended_context = OrderedDict([

            ('cite_context_-1_sent', [citing_sent]),
            ('cite_context_sent_+1', [citing_sent]),
            ('cite_context_-1_sent_+1', [citing_sent]),
            ('cite_context_-2_sent', [citing_sent]),
            ('cite_context_sent_+2', [citing_sent]),
            ('cite_context_-2_sent_+1', [citing_sent]),
            ('cite_context_-1_sent_+2', [citing_sent]),
            ('cite_context_-3_sent', [citing_sent]),
            ('cite_context_sent_+3', [citing_sent])

        ])
        fixed_contexts.append(extended_context)

    else:
        print('fixed context not found!')

    return fixed_contexts

def extract_fixed_context(data_df):

    prev_contexts = list()
    next_contexts = list()
    combined_contexts = list()
    fixed_contexts = list()
    data_df['cite_context_paragraph'] = data_df['cite_context_paragraph'].apply(eval)

    for idx, row in data_df.iterrows():

        context_prev = list()
        context_next = list()
        paragraph = row['cite_context_paragraph']
        citation_context = row['citation_context']

        try:
            citing_sent_index = paragraph.index(citation_context)
            context_prev, context_next = get_prev_next_context(paragraph, citing_sent_index)
        except (IndexError, ValueError) as e:
            print(idx)
            print(citation_context)
            print('Index not found!')

            for index, sent in enumerate(paragraph):
                s = SequenceMatcher(None, sent, citation_context)
                if not s.ratio() > 0.80:
                    continue

                else:
                    paragraph_index = index
                    context_prev, context_next = get_prev_next_context(paragraph, citing_sent_index)
                    break

        fixed_contexts = fixed_prev_next_context(context_prev, citation_context, context_next, fixed_contexts)

    return fixed_contexts

def main():

    parser = argparse.ArgumentParser(description="Code to generate fixed context from paragraph")
    ## Required parameters
    parser.add_argument("--data_set", default='sdp_act', type=str, required=True,
                        help="Dataset used for the experiment (acl_arc or sdp_act)")
    parser.add_argument("--train_size", default=3000, type=int, required=True,
                        help="Length of trainset. sdp_act = 3000 and acl_arc = 1647")

    args = parser.parse_args()
    DATASET_DIR = DATA_PROCESSED_DIR / f"{args.data_set}"
    if not os.path.exists(os.path.join(DATA_PROCESSED_DIR, f'fixed_context_{args.data_set}')):
        os.makedirs(os.path.join(DATA_PROCESSED_DIR, f'fixed_context_{args.data_set}'))

    OUTPUT_DIR = os.path.join(DATA_PROCESSED_DIR, f'fixed_context_{args.data_set}')

    for dataset in ["train", "test"]:
        data_df = pd.read_csv(DATASET_DIR / f"{dataset}.txt", sep="\t", engine="python", dtype=object)
        fixed_contexts = extract_fixed_context(data_df)
        fixed_context_df = pd.DataFrame(fixed_contexts)
        result_df = pd.concat([data_df, fixed_context_df], axis=1)

        if dataset == 'train':
            X_train, X_valid, _, _ = train_test_split(result_df, [0] * args.train_size, test_size=0.166666666,
                                                      random_state=30)
            X_train.to_csv(OUTPUT_DIR + '/train.txt', sep='\t', encoding='utf-8', index=False)
            X_valid.to_csv(OUTPUT_DIR + '/valid.txt', sep='\t', encoding='utf-8', index=False)

        else:
            result_df.to_csv(OUTPUT_DIR + f"/{dataset}.txt", sep='\t', encoding='utf-8',
                             index=False)


if __name__ == '__main__':
    main()


