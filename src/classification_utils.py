import os
import sys
import csv
from nltk.tokenize import sent_tokenize
import json
import ast


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class InputFeaturesText(object):
    """A single set of features of data."""

    def __init__(self, input_ids, label_id):
        self.input_ids = input_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            # reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(cell.unicode('utf-8') for cell in line)
                lines.append(line)
            return lines


class FixedContextCiteSentence(DataProcessor):
    """Processor for fixed context - citing_sentence"""

    def get_train_examples(self, data_dir):
        """see base class"""
        return self._create_examples_train(
            self._read_tsv(os.path.join(data_dir, 'train.txt')), 'train'
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_valid(
            self._read_tsv(os.path.join(data_dir, 'valid.txt')),
            'dev')

    def get_test_examples(self, data_dir):
        """see base class"""
        return self._create_examples_test(
            self._read_tsv(os.path.join(data_dir, 'test.txt')), 'test'
        )

    def get_labels(self):
        """See base class."""
        return ['0', '1', '2', '3', '4', '5']

    def _create_examples_train(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = line[14]
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_valid(self, lines, set_type):
        """Creates examples for validation set."""

        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = line[14]
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_test(self, lines, set_type):
        """Creates examples for test set."""

        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = line[15]
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class FixedContext1Prev(DataProcessor):
    """Processor for fixed context - -1_citing_sentence"""

    def get_train_examples(self, data_dir):
        """see base class"""
        return self._create_examples_train(
            self._read_tsv(os.path.join(data_dir, 'train.txt')), 'train'
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_valid(
            self._read_tsv(os.path.join(data_dir, 'valid.txt')),
            'valid')

    def get_test_examples(self, data_dir):
        """see base class"""
        return self._create_examples_test(
            self._read_tsv(os.path.join(data_dir, 'test.txt')), 'test'
        )

    def get_labels(self):
        """See base class."""
        return ['0', '1', '2', '3', '4', '5']

    def _create_examples_train(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[15]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_valid(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[15]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_test(self, lines, set_type):
        """Creates examples for test set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[16]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class FixedContext1Next(DataProcessor):
    """Processor for fixed context - citing_sentence_+1"""

    def get_train_examples(self, data_dir):
        """see base class"""
        return self._create_examples_train(
            self._read_tsv(os.path.join(data_dir, 'train.txt')), 'train'
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_valid(
            self._read_tsv(os.path.join(data_dir, 'valid.txt')),
            'valid')

    def get_test_examples(self, data_dir):
        """see base class"""
        return self._create_examples_test(
            self._read_tsv(os.path.join(data_dir, 'test.txt')), 'test'
        )

    def get_labels(self):
        """See base class."""
        return ['0', '1', '2', '3', '4', '5']

    def _create_examples_train(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[16]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_valid(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[16]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_test(self, lines, set_type):
        """Creates examples for test set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[17]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class FixedContext1Prev1Next(DataProcessor):
    """Processor for fixed context - -1_citing_sentence_+1"""

    def get_train_examples(self, data_dir):
        """see base class"""
        return self._create_examples_train(
            self._read_tsv(os.path.join(data_dir, 'train.txt')), 'train'
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_valid(
            self._read_tsv(os.path.join(data_dir, 'valid.txt')),
            'valid')

    def get_test_examples(self, data_dir):
        """see base class"""
        return self._create_examples_test(
            self._read_tsv(os.path.join(data_dir, 'test.txt')), 'test'
        )

    def get_labels(self):
        """See base class."""
        return ['0', '1', '2', '3', '4', '5']

    def _create_examples_train(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[17]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_valid(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[17]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_test(self, lines, set_type):
        """Creates examples for test set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[18]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


# change in code

class FixedContext2Prev(DataProcessor):
    """Processor for fixed context - -2_citing_sentence"""

    def get_train_examples(self, data_dir):
        """see base class"""

        return self._create_examples_train(
            self._read_tsv(os.path.join(data_dir, 'train.txt')), 'train'
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_valid(
            self._read_tsv(os.path.join(data_dir, 'valid.txt')),
            'valid')

    def get_test_examples(self, data_dir):
        """see base class"""
        return self._create_examples_test(
            self._read_tsv(os.path.join(data_dir, 'test.txt')), 'test'
        )

    def get_labels(self):
        """See base class."""
        return ['0', '1', '2', '3', '4', '5']

    def _create_examples_train(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[18]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_valid(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[18]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_test(self, lines, set_type):
        """Creates examples for test set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[19]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class FixedContext2Next(DataProcessor):
    """Processor for fixed context - citing_sentence_+2"""

    def get_train_examples(self, data_dir):
        """see base class"""
        return self._create_examples_train(
            self._read_tsv(os.path.join(data_dir, 'train.txt')), 'train'
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_valid(
            self._read_tsv(os.path.join(data_dir, 'valid.txt')),
            'valid')

    def get_test_examples(self, data_dir):
        """see base class"""
        return self._create_examples_test(
            self._read_tsv(os.path.join(data_dir, 'test.txt')), 'test'
        )

    def get_labels(self):
        """See base class."""
        return ['0', '1', '2', '3', '4', '5']

    def _create_examples_train(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[19]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_valid(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[19]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_test(self, lines, set_type):
        """Creates examples for test set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[20]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class FixedContext2Prev1Next(DataProcessor):
    """Processor for fixed context - -2_citing_sentence_+1"""

    def get_train_examples(self, data_dir):
        """see base class"""
        return self._create_examples_train(
            self._read_tsv(os.path.join(data_dir, 'train.txt')), 'train'
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_valid(
            self._read_tsv(os.path.join(data_dir, 'valid.txt')),
            'valid')

    def get_test_examples(self, data_dir):
        """see base class"""
        return self._create_examples_test(
            self._read_tsv(os.path.join(data_dir, 'test.txt')), 'test'
        )

    def get_labels(self):
        """See base class."""
        return ['0', '1', '2', '3', '4', '5']

    def _create_examples_train(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[20]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_valid(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[20]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_test(self, lines, set_type):
        """Creates examples for test set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[21]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class FixedContext1Prev2Next(DataProcessor):
    """Processor for fixed context - -1_citing_sentence_+2"""

    def get_train_examples(self, data_dir):
        """see base class"""
        return self._create_examples_train(
            self._read_tsv(os.path.join(data_dir, 'train.txt')), 'train'
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_valid(
            self._read_tsv(os.path.join(data_dir, 'valid.txt')),
            'valid')

    def get_test_examples(self, data_dir):
        """see base class"""
        return self._create_examples_test(
            self._read_tsv(os.path.join(data_dir, 'test.txt')), 'test'
        )

    def get_labels(self):
        """See base class."""
        return ['0', '1', '2', '3', '4', '5']

    def _create_examples_train(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[21]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_valid(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[21]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_test(self, lines, set_type):
        """Creates examples for test set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[22]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class FixedContext3Prev(DataProcessor):
    """Processor for fixed context - -3_citing_sentence"""

    def get_train_examples(self, data_dir):
        """see base class"""
        return self._create_examples_train(
            self._read_tsv(os.path.join(data_dir, 'train.txt')), 'train'
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_valid(
            self._read_tsv(os.path.join(data_dir, 'valid.txt')),
            'valid')

    def get_test_examples(self, data_dir):
        """see base class"""
        return self._create_examples_test(
            self._read_tsv(os.path.join(data_dir, 'test.txt')), 'test'
        )

    def get_labels(self):
        """See base class."""
        return ['0', '1', '2', '3', '4', '5']

    def _create_examples_train(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[22]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_valid(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[22]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_test(self, lines, set_type):
        """Creates examples for test set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[23]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class FixedContext3Next(DataProcessor):
    """Processor for fixed context - citing_sentence_+3"""

    def get_train_examples(self, data_dir):
        """see base class"""
        return self._create_examples_train(
            self._read_tsv(os.path.join(data_dir, 'train.txt')), 'train'
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_valid(
            self._read_tsv(os.path.join(data_dir, 'valid.txt')),
            'valid')

    def get_test_examples(self, data_dir):
        """see base class"""
        return self._create_examples_test(
            self._read_tsv(os.path.join(data_dir, 'test.txt')), 'test'
        )

    def get_labels(self):
        """See base class."""
        return ['0', '1', '2', '3', '4', '5']

    def _create_examples_train(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[23]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_valid(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[23]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_test(self, lines, set_type):
        """Creates examples for test set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = "\n".join(ast.literal_eval(line[24]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class FixedContextParagraph(DataProcessor):
    """Processor for fixed context citing_sentence_paragraph"""

    def get_train_examples(self, data_dir):
        """see base class"""
        return self._create_examples_train(
            self._read_tsv(os.path.join(data_dir, 'train.txt')), 'train'
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_valid(
            self._read_tsv(os.path.join(data_dir, 'valid.txt')),
            'valid')

    def get_test_examples(self, data_dir):
        """see base class"""
        return self._create_examples_test(
            self._read_tsv(os.path.join(data_dir, 'test.txt')), 'test'
        )

    def get_labels(self):
        """See base class."""
        return ['0', '1', '2', '3', '4', '5']

    def _create_examples_train(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            # print(json.loads(line[16]))
            text_a = "\n".join(ast.literal_eval(line[8]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_valid(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            # print(json.loads(line[16]))
            text_a = "\n".join(ast.literal_eval(line[8]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_test(self, lines, set_type):
        """Creates examples for test set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            # line[14] = ast.literal_eval(line[14])
            text_a = "\n".join(ast.literal_eval(line[8]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class DynamicContextSpecterOne(DataProcessor):
    """Processor for Dynamic combined context using PDTB"""

    def get_train_examples(self, data_dir):
        """see base class"""
        return self._create_examples_train(
            self._read_tsv(os.path.join(data_dir, 'train.txt')), 'train'
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_valid(
            self._read_tsv(os.path.join(data_dir, 'valid.txt')),
            'valid')

    def get_test_examples(self, data_dir):
        """see base class"""
        return self._create_examples_test(
            self._read_tsv(os.path.join(data_dir, 'test.txt')), 'test'
        )

    def get_labels(self):
        """See base class."""
        return ['0', '1', '2', '3', '4', '5']

    def _create_examples_train(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = "\n".join(ast.literal_eval(line[24]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_valid(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            # print(json.loads(line[16]))
            text_a = "\n".join(ast.literal_eval(line[24]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_test(self, lines, set_type):
        """Creates examples for test set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            # line[14] = ast.literal_eval(line[14])
            text_a = "\n".join(ast.literal_eval(line[26]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class DynamicContextSpecterTwo(DataProcessor):
    """Processor for Dynamic combined context using PDTB"""

    def get_train_examples(self, data_dir):
        """see base class"""
        return self._create_examples_train(
            self._read_tsv(os.path.join(data_dir, 'train.txt')), 'train'
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_valid(
            self._read_tsv(os.path.join(data_dir, 'valid.txt')),
            'valid')

    def get_test_examples(self, data_dir):
        """see base class"""
        return self._create_examples_test(
            self._read_tsv(os.path.join(data_dir, 'test.txt')), 'test'
        )

    def get_labels(self):
        """See base class."""
        return ['0', '1', '2', '3', '4', '5']

    def _create_examples_train(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = "\n".join(ast.literal_eval(line[26]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_valid(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            # print(json.loads(line[16]))
            text_a = "\n".join(ast.literal_eval(line[26]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_test(self, lines, set_type):
        """Creates examples for test set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            # line[14] = ast.literal_eval(line[14])
            text_a = "\n".join(ast.literal_eval(line[27]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class DynamicContextSpecterThree(DataProcessor):
    """Processor for Dynamic combined context using PDTB"""

    def get_train_examples(self, data_dir):
        """see base class"""
        return self._create_examples_train(
            self._read_tsv(os.path.join(data_dir, 'train.txt')), 'train'
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_valid(
            self._read_tsv(os.path.join(data_dir, 'valid.txt')),
            'valid')

    def get_test_examples(self, data_dir):
        """see base class"""
        return self._create_examples_test(
            self._read_tsv(os.path.join(data_dir, 'test.txt')), 'test'
        )

    def get_labels(self):
        """See base class."""
        return ['0', '1', '2', '3', '4', '5']

    def _create_examples_train(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = "\n".join(ast.literal_eval(line[28]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_valid(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            # print(json.loads(line[16]))
            text_a = "\n".join(ast.literal_eval(line[28]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_test(self, lines, set_type):
        """Creates examples for test set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            # line[14] = ast.literal_eval(line[14])
            text_a = "\n".join(ast.literal_eval(line[29]))
            text_b = ""
            label = line[9]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class DynamicContextConnectorPrev(DataProcessor):
    """Processor for Dynamic prev context using Explicit connector"""

    def get_train_examples(self, data_dir):

        return self._create_examples_train(
            self._read_tsv(os.path.join(data_dir, 'train.txt')), 'train'
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_valid(
            self._read_tsv(os.path.join(data_dir, 'valid.txt')),
            'valid')

    def get_test_examples(self, data_dir):
        """see base class"""
        return self._create_examples_test(
            self._read_tsv(os.path.join(data_dir, 'test.txt')), 'test'
        )

    def get_labels(self):
        """See base class."""
        return ['0', '1', '2', '3', '4', '5']

    def _create_examples_train(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = "\n".join(ast.literal_eval(line[23]))
            text_b = ""
            label = line[21]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_valid(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            # print(json.loads(line[16]))
            text_a = "\n".join(ast.literal_eval(line[23]))
            text_b = ""
            label = line[21]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_test(self, lines, set_type):
        """Creates examples for test set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            # line[14] = ast.literal_eval(line[14])
            text_a = "\n".join(ast.literal_eval(line[23]))
            text_b = ""
            label = line[21]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class DynamicContextConnectorNext(DataProcessor):

    def get_train_examples(self, data_dir):
        """see base class"""
        return self._create_examples_train(
            self._read_tsv(os.path.join(data_dir, 'train.txt')), 'train'
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_valid(
            self._read_tsv(os.path.join(data_dir, 'valid.txt')),
            'valid')

    def get_test_examples(self, data_dir):
        """see base class"""
        return self._create_examples_test(
            self._read_tsv(os.path.join(data_dir, 'test.txt')), 'test'
        )

    def get_labels(self):
        """See base class."""
        return ['0', '1', '2', '3', '4', '5']

    def _create_examples_train(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = "\n".join(ast.literal_eval(line[25]))
            text_b = ""
            label = line[21]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_valid(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            # print(json.loads(line[16]))
            text_a = "\n".join(ast.literal_eval(line[25]))
            text_b = ""
            label = line[21]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_test(self, lines, set_type):
        """Creates examples for test set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            # line[14] = ast.literal_eval(line[14])
            text_a = "\n".join(ast.literal_eval(line[25]))
            text_b = ""
            label = line[21]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class DynamicContextConnectorCombined(DataProcessor):
    """Processor for Dynamic combined context using explicit connector"""

    def get_train_examples(self, data_dir):
        """see base class"""
        return self._create_examples_train(
            self._read_tsv(os.path.join(data_dir, 'train.txt')), 'train'
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_valid(
            self._read_tsv(os.path.join(data_dir, 'valid.txt')),
            'valid')

    def get_test_examples(self, data_dir):
        """see base class"""
        return self._create_examples_test(
            self._read_tsv(os.path.join(data_dir, 'test.txt')), 'test'
        )

    def get_labels(self):
        """See base class."""
        return ['0', '1', '2', '3', '4', '5']

    def _create_examples_train(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = "\n".join(ast.literal_eval(line[27]))
            text_b = ""
            label = line[21]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_valid(self, lines, set_type):
        """Creates examples for training set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            # print(json.loads(line[16]))
            text_a = "\n".join(ast.literal_eval(line[27]))
            text_b = ""
            label = line[21]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_examples_test(self, lines, set_type):
        """Creates examples for test set."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            # line[14] = ast.literal_eval(line[14])
            text_a = "\n".join(ast.literal_eval(line[27]))
            text_b = ""
            label = line[21]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, print_examples=False):
    """
    Loads a data file into a list of InputBatch objects
    :param examples:
    :param max_seq_length:
    :param tokenizer:
    :param print_examples:
    :return: a list of InputBatch objects
    """
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        # Replacing new lines with [SEP] tokens
        text_a = example.text_a.replace('\n', '[SEP]')
        tokens_a = tokenizer.tokenize(text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # label_id = [float(x) for x in example.label]
        label_id = label_map[example.label]

        if print_examples and ex_index < 5:
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s" % example.label)

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=label_id))
    return features


def convert_examples_to_hierarchical_features(examples, max_seq_length, tokenizer, print_examples=False):
    """
    Loads a data file into a list of InputBatch objects
    :param examples:
    :param max_seq_length:
    :param tokenizer:
    :param print_examples:
    :return: a list of InputBatch objects
    """
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = [tokenizer.tokenize(line) for line in sent_tokenize(example.text_a)]
        tokens_b = None

        if example.text_b:
            tokens_b = [tokenizer.tokenize(line) for line in sent_tokenize(example.text_b)]
            # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length
            # Account for [CLS], [SEP], [SEP]
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP]
            for i0 in range(len(tokens_a)):
                if len(tokens_a[i0]) > max_seq_length - 2:
                    tokens_a[i0] = tokens_a[i0][:(max_seq_length - 2)]

        tokens = [["[CLS]"] + line + ["[SEP]"] for line in tokens_a]
        segment_ids = [[0] * len(line) for line in tokens]

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = list()
        for line in tokens:
            input_ids.append(tokenizer.convert_tokens_to_ids(line))

        # Input mask has 1 for real tokens and 0 for padding tokens
        input_mask = [[1] * len(line_ids) for line_ids in input_ids]

        # Zero-pad up to the sequence length.
        padding = [[0] * (max_seq_length - len(line_ids)) for line_ids in input_ids]
        for i0 in range(len(input_ids)):
            input_ids[i0] += padding[i0]
            input_mask[i0] += padding[i0]
            segment_ids[i0] += padding[i0]

        # label_id = [float(x) for x in example.label]
        label_id = label_map[example.label]

        if print_examples and ex_index < 5:
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s" % example.label)

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
    Truncates a sequence pair in place to the maximum length
    :param tokens_a:
    :param tokens_b:
    :param max_length:
    :return:
    """

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


output_modes = {
    "citation_function": "classification",
    "citation_influence": "classification",
}

CITATION_CLASSIFICATION_NUM_LABELS = {
    "citation_function": 6,
    "citation_influence": 2
}

processors_fixed_context = {
    "cite_sent": FixedContextCiteSentence,
    "prev1_cite_sent": FixedContext1Prev,
    "cite_sent_next1": FixedContext1Next,
    "prev1_cite_sent_next1": FixedContext1Prev1Next,
    "prev2_cite_sent": FixedContext2Prev,
    "cite_sent_next2": FixedContext2Next,
    "prev2_cite_sent_next1": FixedContext2Prev1Next,
    "prev1_cite_sent_next2": FixedContext1Prev2Next,
    "prev3_cite_sent": FixedContext3Prev,
    "cite_sent_next3": FixedContext3Next,
    "cite_sent_para": FixedContextParagraph
}

processors_dynamic_context = {

    "specter_context_one": DynamicContextSpecterOne,
    "specter_context_two": DynamicContextSpecterTwo,
    "specter_context_three": DynamicContextSpecterThree,
    "explicit_connector_prev": DynamicContextConnectorPrev,
    "explicit_connector_next": DynamicContextConnectorNext,
    "explicit_connector_combined": DynamicContextConnectorCombined
}