from difflib import SequenceMatcher
from dynamic_context_utils import get_prev_next_context, compute_smilarity, tokenize
from src.dynamic_context import TOKENIZER


class InputFeatures(object):

    def __init__(self, citing_title, cited_title, citation_context, cited_abstract, paragraph):
        self.citing_title = citing_title
        self.cited_title = cited_title
        self.citation_context = citation_context
        self.cited_abstract = cited_abstract
        self.paragraph = paragraph


class ContiguousContextExp1(InputFeatures):
    """A single set of features of data."""

    def extract_embeddings(self):

        title_abs = [self.cited_title + TOKENIZER.sep_token + self.cited_abstract]
        input_seq = [self.citing_title + TOKENIZER.sep_token + self.citation_context]
        citing_context_emb = tokenize(input_seq)
        cited_abstract_emb = tokenize(title_abs)
        similarity_citation_context = compute_smilarity(cited_abstract_emb, citing_context_emb)
        return cited_abstract_emb, similarity_citation_context

    def extract_context(self, cited_abstract_emb, similarity_citation_context):

        try:
            citing_sent_index = self.paragraph.index(self.citation_context)
            prev, next = get_prev_next_context(self.paragraph, citing_sent_index)
            dynamic_context_prev = self.extract_context_from_paragraph(prev, similarity_citation_context,
                                                                       self.citing_title,
                                                                       cited_abstract_emb) + [self.citation_context]
            dynamic_context_next = [self.citation_context] + self.extract_context_from_paragraph(next,
                                                                                                 similarity_citation_context,
                                                                                                 self.citing_title,
                                                                                                 cited_abstract_emb)
            dynamic_context_combined = self.extract_context_from_paragraph(prev, similarity_citation_context,
                                                                           self.citing_title,
                                                                           cited_abstract_emb) + [
                                           self.citation_context] + \
                                       self.extract_context_from_paragraph(next, similarity_citation_context,
                                                                           self.citing_title, cited_abstract_emb)
        except (IndexError, ValueError):
            for i, sent in enumerate(self.paragraph):
                s = SequenceMatcher(None, self.citation_context, sent)
                if s.ratio() > 0.80:
                    citing_sent_index = i
                    prev, next = get_prev_next_context(self.paragraph, citing_sent_index)
                    dynamic_context_prev = self.extract_context_from_paragraph(prev, similarity_citation_context,
                                                                               self.citing_title,
                                                                               cited_abstract_emb) + [self.citation_context]
                    dynamic_context_next = [self.citation_context] + self.extract_context_from_paragraph(next,
                                                                                                         similarity_citation_context,
                                                                                                         self.citing_title,
                                                                                                         cited_abstract_emb)

                    dynamic_context_combined = self.extract_context_from_paragraph(prev, similarity_citation_context,
                                                                                   self.citing_title,
                                                                                   cited_abstract_emb) + [
                                                   self.citation_context] + \
                                               self.extract_context_from_paragraph(next, similarity_citation_context,
                                                                                   self.citing_title,
                                                                                   cited_abstract_emb)

                    break
                else:
                    continue

            if citing_sent_index is None:
                dynamic_context_prev = [self.citation_context]
                dynamic_context_next = [self.citation_context]
                dynamic_context_combined = [self.citation_context]

        return dynamic_context_prev, dynamic_context_next, dynamic_context_combined

    def extract_context_from_paragraph(self, context, similarity_citation_context, citing_title, cited_abstract_emb):
        dynamic_context = list()

        if context:
            for sent in context:
                para_sent_seq = [citing_title + TOKENIZER.sep_token + sent]
                para_sent_emb = tokenize(para_sent_seq)
                similarity_sent = compute_smilarity(cited_abstract_emb, para_sent_emb)
                if similarity_sent >= similarity_citation_context:
                    dynamic_context.append(sent)

                else:
                    break

        return dynamic_context


class ContiguousContextExp2(InputFeatures):
    """A single set of features of data."""

    def extract_embeddings(self):

        title_abs = [self.cited_title + TOKENIZER.sep_token + '']
        input_seq = ['' + TOKENIZER.sep_token + self.citation_context]
        citing_context_emb = tokenize(input_seq)
        cited_abstract_emb = tokenize(title_abs)
        similarity_citation_context = compute_smilarity(cited_abstract_emb, citing_context_emb)
        return cited_abstract_emb, similarity_citation_context

    def extract_context(self, cited_abstract_emb, similarity_citation_context):

        try:
            citing_sent_index = self.paragraph.index(self.citation_context)
            prev, next = get_prev_next_context(self.paragraph, citing_sent_index)
            dynamic_context_prev = self.extract_context_from_paragraph(prev, similarity_citation_context,
                                                                 cited_abstract_emb) + [self.citation_context]
            dynamic_context_next = [self.citation_context] + self.extract_context_from_paragraph(next, similarity_citation_context,
                                                                                      cited_abstract_emb)

            dynamic_context_combined = self.extract_context_from_paragraph(prev, similarity_citation_context,
                                                                     cited_abstract_emb) \
                                       + [self.citation_context] + \
                                       self.extract_context_from_paragraph(next, similarity_citation_context,
                                                                     cited_abstract_emb)
        except (IndexError, ValueError):
            for i, sent in enumerate(self.paragraph):
                s = SequenceMatcher(None, self.citation_context, sent)
                if s.ratio() > 0.80:
                    citing_sent_index = i
                    prev, next = get_prev_next_context(self.paragraph, citing_sent_index)
                    dynamic_context_prev = self.extract_context_from_paragraph(prev, similarity_citation_context,
                                                                         cited_abstract_emb) + [self.citation_context]
                    dynamic_context_next = [self.citation_context] + self.extract_context_from_paragraph(prev,
                                                                                              similarity_citation_context,
                                                                                              cited_abstract_emb)

                    dynamic_context_combined = self.extract_context_from_paragraph(prev, similarity_citation_context,
                                                                             cited_abstract_emb) + [self.citation_context] + \
                                               self.extract_context_from_paragraph(prev, similarity_citation_context,
                                                                             cited_abstract_emb)

                    break
                else:
                    continue

            if citing_sent_index is None:
                dynamic_context_prev = [self.citation_context]
                dynamic_context_next = [self.citation_context]
                dynamic_context_combined = [self.citation_context]

        return dynamic_context_prev, dynamic_context_next, dynamic_context_combined

    def extract_context_from_paragraph(self, context, similarity_citation_context, cited_abstract_emb):

        dynamic_context = list()

        if context:
            for sent in context:
                para_sent_seq = ['' + TOKENIZER.sep_token + sent]
                para_sent_emb = tokenize(para_sent_seq)
                similarity_sent = compute_smilarity(cited_abstract_emb, para_sent_emb)
                if similarity_sent >= similarity_citation_context:
                    dynamic_context.append(sent)

                else:
                    break

        return dynamic_context


class ContiguousContextExp3(InputFeatures):

    def extract_embeddings(self):

        title_abs = [self.cited_title + TOKENIZER.sep_token + self.cited_abstract]
        input_seq = ['' + TOKENIZER.sep_token + self.citation_context]
        citing_context_emb = tokenize(input_seq)
        cited_abstract_emb = tokenize(title_abs)
        similarity_citation_context = compute_smilarity(cited_abstract_emb, citing_context_emb)
        return cited_abstract_emb, similarity_citation_context

    def extract_context(self, cited_abstract_emb, similarity_citation_context):

        try:
            citing_sent_index = self.paragraph.index(self.citation_context)
            prev, next = get_prev_next_context(self.paragraph, citing_sent_index)
            dynamic_context_prev = self.extract_context_from_paragraph(prev, similarity_citation_context,
                                                                     cited_abstract_emb) + [self.citation_context]
            dynamic_context_next = [self.citation_context] + self.extract_context_from_paragraph(next,
                                                                                          similarity_citation_context,
                                                                                          cited_abstract_emb)

            dynamic_context_combined = self.extract_context_from_paragraph(prev, similarity_citation_context,
                                                                         cited_abstract_emb) \
                                        + [self.citation_context] + \
                                        self.extract_context_from_paragraph(next, similarity_citation_context,
                                                                         cited_abstract_emb)

        except (IndexError, ValueError) as e:

            for i, sent in enumerate(self.paragraph):
                s = SequenceMatcher(None, self.citation_context, sent)
                if s.ratio() > 0.80:
                    citing_sent_index = i
                    prev, next = get_prev_next_context(self.paragraph, citing_sent_index)
                    dynamic_context_prev = self.extract_context_from_paragraph(prev, similarity_citation_context,
                                                                             cited_abstract_emb) + [self.citation_context]
                    dynamic_context_next = [self.citation_context] + self.extract_context_from_paragraph(prev,
                                                                                                  similarity_citation_context,
                                                                                                  cited_abstract_emb)
                    dynamic_context_combined = self.extract_context_from_paragraph(prev, similarity_citation_context,
                                                                            cited_abstract_emb) + [
                                                    self.citation_context] + \
                                                self.extract_context_from_paragraph(prev, similarity_citation_context,
                                                                                 cited_abstract_emb)

                    break
                else:
                    continue

            if citing_sent_index is None:
                dynamic_context_prev = [self.citation_context]
                dynamic_context_next = [self.citation_context]
                dynamic_context_combined = [self.citation_context]

        return dynamic_context_prev, dynamic_context_next, dynamic_context_combined

    def extract_context_from_paragraph(self, context, similarity_citation_context, cited_abstract_emb):

        dynamic_context = list()
        if context:
            for sent in context:
                para_sent_seq = ['' + TOKENIZER.sep_token + sent]
                para_sent_emb = tokenize(para_sent_seq)
                similarity_sent = compute_smilarity(cited_abstract_emb, para_sent_emb)
                if similarity_sent >= similarity_citation_context:
                    dynamic_context.append(sent)

                else:
                    break

        return dynamic_context


class ContiguousContextExp4(InputFeatures):

    def extract_embeddings(self):

        title_abs = [self.cited_title + TOKENIZER.sep_token + self.cited_abstract]
        input_seq = [self.cited_title + TOKENIZER.sep_token + self.citation_context]
        citing_context_emb = tokenize(input_seq)
        cited_abstract_emb = tokenize(title_abs)
        similarity_citation_context = compute_smilarity(cited_abstract_emb, citing_context_emb)
        return cited_abstract_emb, similarity_citation_context

    def extract_context(self, cited_abstract_emb, similarity_citation_context):

        try:
            citing_sent_index = self.paragraph.index(self.citation_context)
            prev, next = get_prev_next_context(self.paragraph, citing_sent_index)
            dynamic_context_prev = self.extract_context_from_paragraph(prev, similarity_citation_context, self.cited_title,
                                                                 cited_abstract_emb) + [self.citation_context]
            dynamic_context_next = [self.citation_context] + self.extract_context_from_paragraph(next, similarity_citation_context,
                                                                                      self.cited_title, cited_abstract_emb)

            dynamic_context_combined = self.extract_context_from_paragraph(prev, similarity_citation_context, self.cited_title,
                                                                     cited_abstract_emb) + [self.citation_context] + \
                                       self.extract_context_from_paragraph(next, similarity_citation_context,
                                                                     self.cited_title, cited_abstract_emb)

        except (IndexError, ValueError) as e:

            for i, sent in enumerate(self.paragraph):
                s = SequenceMatcher(None, self.citation_context, sent)
                if s.ratio() > 0.80:
                    citing_sent_index = i
                    prev, next = get_prev_next_context(self.paragraph, citing_sent_index)
                    dynamic_context_prev = self.extract_context_from_paragraph(prev, similarity_citation_context,
                                                                         self.citing_title,
                                                                         cited_abstract_emb) + [self.citation_context]
                    dynamic_context_next = [self.citation_context] + self.extract_context_from_paragraph(next,
                                                                                              similarity_citation_context,
                                                                                              self.citing_title,
                                                                                              cited_abstract_emb)

                    dynamic_context_combined = self.extract_context_from_paragraph(prev, similarity_citation_context,
                                                                             self.citing_title,
                                                                             cited_abstract_emb) + [self.citation_context] + \
                                               self.extract_context_from_paragraph(next, similarity_citation_context,
                                                                             self.citing_title, cited_abstract_emb)

                    break
                else:
                    continue

            if citing_sent_index is None:
                dynamic_context_prev = [self.citation_context]
                dynamic_context_next = [self.citation_context]
                dynamic_context_combined = [self.citation_context]

        return dynamic_context_prev, dynamic_context_next, dynamic_context_combined

    def extract_context_from_paragraph(self, context, similarity_citation_context, cited_abstract_emb):

        dynamic_context = list()
        if context:
            for sent in context:
                para_sent_seq = [self.cited_title + TOKENIZER.sep_token + sent]
                para_sent_emb = tokenize(para_sent_seq)
                similarity_sent = compute_smilarity(cited_abstract_emb, para_sent_emb)
                if similarity_sent >= similarity_citation_context:
                    dynamic_context.append(sent)

                else:
                    break

        return dynamic_context


processors_dynamic_context_contiguous = {

    "exp1": ContiguousContextExp1,
    "exp2": ContiguousContextExp2,
    "exp3": ContiguousContextExp3,
    "exp4": ContiguousContextExp4,

}
