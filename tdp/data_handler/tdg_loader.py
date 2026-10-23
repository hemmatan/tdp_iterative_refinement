import logging
"""Read gold-node TDG documents and construct model inputs.

Corpus edges are written child-to-parent; internal matrices and DGL edges
use parent-to-child ordering. ROOT insertion shifts sentence indices by one."""

import re
import dgl
import torch
from random import shuffle
import csv
import os
from transformers import AutoTokenizer
from tdp.data_handler.utils import Vocabulary
from tdp.data_handler.data_transformation import (
    add_root_and_reindex,
)
from graph4nlp.pytorch.data.data import GraphData
import tdp.constants as constants


class BasicDataLoader:
    DEPID_FEATURE = "_depID"
    TOKID_FEATURE = "_tokID"

    def __init__(
        self,
        filename=None,
        token_vocabulary=None,
        edge_vocabulary=None,
        apply_algebra=False,
        encoder_type=None,
    ):
        assert filename is not None, "Filename must be provided to BasicDataLoader"
        assert (
            encoder_type is not None
        ), "Encoder type must be specified (e.g., 'BERT' or 'ROBERTA')"
        self.example_loader = TDG_file_reader(filename, encoder_type=encoder_type)
        if apply_algebra:
            raise ValueError("The paper protocol does not augment temporal edges.")
        if token_vocabulary:
            self.tok_vocab = token_vocabulary
        else:
            self.tok_vocab = Vocabulary()
            self.tok_vocab.from_dictionary(self.example_loader.tokenizer.get_vocab())
        if edge_vocabulary:
            self.edge_vocabulary = edge_vocabulary
        else:
            self.edge_vocabulary = self.example_loader.edge_vocab
        self.graphs = [
            self.example_to_dgl(example) for example in self.example_loader.dataset
        ]
        self.dataset_stats = self.get_dataset_stats()

    def example_to_dgl(self, example):
        esrc = [
            example.REF_GRAPH.edge_attributes[i]["src"]
            for i in range(len(example.REF_GRAPH.edge_attributes))
        ]
        edst = [
            example.REF_GRAPH.edge_attributes[i]["tgt"]
            for i in range(len(example.REF_GRAPH.edge_attributes))
        ]
        elbl = [
            example.REF_GRAPH.edge_attributes[i]["label"]
            for i in range(len(example.REF_GRAPH.edge_attributes))
        ]
        depgraph = dgl.graph((torch.tensor(esrc), torch.tensor(edst)))
        depgraph.edata[BasicDataLoader.DEPID_FEATURE] = torch.tensor(
            [self.edge_vocabulary(e) for e in elbl]
        )
        depgraph.ndata[BasicDataLoader.TOKID_FEATURE] = torch.tensor(
            [example.NODE_INTERVALS2ID[tok] for tok in example.NODE_INTERVALS2ID]
        )
        return depgraph


    def generate_batch(self, batch_size, randomize=True):
        orderlist = list(range(len(self.graphs)))
        if randomize:
            shuffle(orderlist)
        newlist = [self.graphs[i] for i in orderlist]
        bs = 0
        while bs < len(newlist):
            be = min(len(newlist), bs + batch_size)
            yield (dgl.batch(newlist[bs:be]), orderlist[bs:be])
            bs = be


    def get_dataset_stats(self):
        number_of_examples = len(self.example_loader.dataset)
        vocabulary_size = len(self.tok_vocab)
        edge_vocabulary_size = len(self.edge_vocabulary)
        max_nodes = max((graph.num_nodes() for graph in self.graphs))
        min_nodes = min((graph.num_nodes() for graph in self.graphs))
        max_edges = max((graph.num_edges() for graph in self.graphs))
        min_edges = min((graph.num_edges() for graph in self.graphs))
        avg_nodes = (
            sum((graph.num_nodes() for graph in self.graphs)) / number_of_examples
        )
        avg_edges = (
            sum((graph.num_edges() for graph in self.graphs)) / number_of_examples
        )
        node_type_counts = {}
        for example in self.example_loader.dataset:
            for node_data in example.REF_GRAPH.node_attributes:
                node_type = node_data["temporal_tag"]
                if node_type not in node_type_counts:
                    node_type_counts[node_type] = 0
                node_type_counts[node_type] += 1
        edge_type_counts = {}
        for example in self.example_loader.dataset:
            for edge_data in example.REF_GRAPH.edge_attributes:
                edge_type = edge_data["label"]
                if edge_type not in edge_type_counts:
                    edge_type_counts[edge_type] = 0
                edge_type_counts[edge_type] += 1
        max_sentences = max(
            (len(example.TOKENS) for example in self.example_loader.dataset)
        )
        min_sentences = min(
            (len(example.TOKENS) for example in self.example_loader.dataset)
        )
        return {
            "number_of_examples": number_of_examples,
            "vocabulary_size": vocabulary_size,
            "edge_vocabulary_size": edge_vocabulary_size,
            "max_nodes": max_nodes,
            "min_nodes": min_nodes,
            "max_edges": max_edges,
            "min_edges": min_edges,
            "avg_nodes": avg_nodes,
            "avg_edges": avg_edges,
            "node_type_counts": node_type_counts,
            "edge_type_counts": edge_type_counts,
            "max_sentences": max_sentences,
            "min_sentences": min_sentences,
        }


class TAGencoder:

    def __init__(self, tagset):
        self.tag2id = dict(((tag, idx) for idx, tag in enumerate(tagset)))
        self.id2tag = tagset


class Example:

    def __init__(
        self,
        tokenizer,
        text_string,
        dataset,
        nodelist=None,
        taglist=None,
        govlist=None,
        edgelist=None,
        root_token=None,
        maxlen=constants.MAXLEN_BERT,
        stride=constants.STRIDE_BERT,
        doc_id=-1,
    ):
        if nodelist is None:
            nodelist = []
        assert type(text_string) == list
        assert len(nodelist) == len(taglist)
        assert 0 < stride <= maxlen
        assert set(govlist) <= set(nodelist + ["-1_-1_-1"])
        self.DATASET = dataset
        self.DOC_ID = doc_id
        self.tokenizer = tokenizer
        if root_token:
            text_string, nodelist, govlist = add_root_and_reindex(
                text_string, nodelist, govlist, root_token
            )
        self.TOKENS = [tokenizer.tokenize(sentence) for sentence in text_string]
        self.NTOKENS = sum((len(line) for line in self.TOKENS))
        self.TOKENS_LIST = tokenizer.tokenize(" ".join(text_string))
        self.tokens2sub = self.tokens2pieces(text_string, tokenizer)
        self.TOKENS_IDS, self.TOKENS_SEGMENTATION = self.sliding_window_tokenization(
            tokenizer, " ".join(text_string), maxlen, stride
        )
        self.node_interval2subword_idx = {}
        if nodelist:
            nodeset, tagset = zip(*list(dict.fromkeys(zip(nodelist, taglist))))
            nodeset = nodeset + ("0_0_0",)
            tagset = tagset + ("ROOT",)
            self.REF_GRAPH = GraphData()
            self.REF_GRAPH.add_nodes(len(nodeset))
            for node_idx in range(len(nodeset)):
                self.REF_GRAPH.node_attributes[node_idx]["node_id"] = nodeset[node_idx]
                self.REF_GRAPH.node_attributes[node_idx]["subword_idxes"] = (
                    self.node2posn(nodeset[node_idx])
                )
                self.REF_GRAPH.node_attributes[node_idx]["temporal_tag"] = tagset[
                    node_idx
                ]
            for attr in self.REF_GRAPH.node_attributes:
                self.node_interval2subword_idx[attr["node_id"]] = attr["subword_idxes"]
            self.TEMPORAL_TAG_ENCODER = TAGencoder(tagset=["Event", "Timex", "ROOT"])
            temporalIDs = [self.TEMPORAL_TAG_ENCODER.tag2id[tag] for tag in tagset]
            self.REF_GRAPH.node_features["ref_times"] = torch.tensor(temporalIDs)
            biocoder = TAGencoder(tagset=["B", "I", "O"])
            self.REF_BIO = [biocoder.tag2id["O"]] * self.NTOKENS
            for node_idx in range(len(nodeset)):
                current_tag = biocoder.tag2id["B"]
                for subword_idx in self.REF_GRAPH.node_attributes[node_idx][
                    "subword_idxes"
                ]:
                    self.REF_BIO[subword_idx] = current_tag
                    current_tag = biocoder.tag2id["I"]
            self.NODE_INTERVALS2ID = dict(
                ((node, idx) for idx, node in enumerate(nodeset))
            )
            self.NODE_ID2INTERVAL = dict(
                ((idx, node) for idx, node in enumerate(nodeset))
            )
            self.REF_LINKS = torch.zeros(len(nodeset), len(nodeset))
            self.edge_ids2labels = {}
            for i in range(len(edgelist)):
                src = self.NODE_INTERVALS2ID[govlist[i]]
                tgt = self.NODE_INTERVALS2ID[nodelist[i]]
                self.add_single_edge(src, tgt, edgelist[i], self.DATASET)
            self.node_ids2labels = {}
            for i in range(len(self.NODE_INTERVALS2ID)):
                subword_idxes = self.REF_GRAPH.node_attributes[i]["subword_idxes"]
                self.node_ids2labels[i] = " ".join(
                    self.TOKENS_LIST[
                        subword_idxes[0] : subword_idxes[len(subword_idxes) - 1] + 1
                    ]
                )

    def add_single_edge(
        self, src, tgt, label, dataset, depth=0, operation=None, visited=None
    ):
        failures_csv_file = "algebra_failures.csv"
        success_csv_file = "algebra_success.csv"
        node_h_str = self.node_id2string(src)
        node_d_str = self.node_id2string(tgt)
        if (src, tgt) in self.edge_ids2labels:
            same_label = label == self.edge_ids2labels[src, tgt]
            existing_depth = next(
                (
                    item["detection_time"]
                    for item in self.REF_GRAPH.edge_attributes
                    if item.get("src") == src and item.get("tgt") == tgt
                ),
                None,
            )
            failure = {
                "src": self.NODE_ID2INTERVAL[src],
                "src_str": node_h_str,
                "tgt": self.NODE_ID2INTERVAL[tgt],
                "tgt_str": node_d_str,
                "existing_label": self.edge_ids2labels[src, tgt],
                "new_label": label,
                "same_label": same_label,
                "depth": depth,
                "previous_depth": existing_depth.item(),
                "operation": operation,
                "text": self.TOKENS,
                "dataset": dataset,
            }
            file_exists_and_not_empty = (
                os.path.exists(failures_csv_file)
                and os.path.getsize(failures_csv_file) > 0
            )
            with open(
                failures_csv_file, mode="a", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(file, fieldnames=failure.keys())
                if not file_exists_and_not_empty:
                    writer.writeheader()
                writer.writerow(failure)
            return False
        self.REF_LINKS[src, tgt] = 1
        self.REF_GRAPH.add_edge(src, tgt)
        self.REF_GRAPH.edge_attributes[-1]["label"] = label
        self.REF_GRAPH.edge_attributes[-1]["src"] = src
        self.REF_GRAPH.edge_attributes[-1]["tgt"] = tgt
        self.REF_GRAPH.edge_attributes[-1]["detection_time"] = torch.tensor(depth)
        self.edge_ids2labels[src, tgt] = label
        if operation:
            new_edge = {
                "src": self.NODE_ID2INTERVAL[src],
                "src_str": node_h_str,
                "tgt": self.NODE_ID2INTERVAL[tgt],
                "tgt_str": node_d_str,
                "label": label,
                "depth": depth,
                "operation": operation,
                "text": self.TOKENS,
            }
            file_exists_and_not_empty = (
                os.path.exists(success_csv_file)
                and os.path.getsize(success_csv_file) > 0
            )
            with open(success_csv_file, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=new_edge.keys())
                if not file_exists_and_not_empty:
                    writer.writeheader()
                writer.writerow(new_edge)
        return True


    def node_id2string(self, node_id):
        node_str = ""
        subword_idxes = self.REF_GRAPH.node_attributes[node_id]["subword_idxes"]
        token_list = self.TOKENS_LIST[subword_idxes[0] : subword_idxes[-1] + 1]
        for i, token in enumerate(token_list):
            if token.startswith("##"):
                node_str += token.replace("##", "")
            elif i == 0:
                node_str += token
            else:
                node_str += " " + token
        return node_str

    def node_id2type(self, node_id):
        return self.REF_GRAPH.node_attributes[node_id]["temporal_tag"]

    def node_id2position(self, node_id):
        return self.REF_GRAPH.node_attributes[node_id]["subword_idxes"]

    def node_id2sentence_number(self, node_id):
        return int(self.REF_GRAPH.node_attributes[node_id]["node_id"].split("_")[0])

    def node_id2no_of_tokens(self, node_id):
        return len(self.REF_GRAPH.node_attributes[node_id]["subword_idxes"])

    def get_context_between(self, s, e):
        context = []
        for i in range(min(s, e), max(s, e) + 1):
            context.append(self.TOKENS[i])
        return context

    def sliding_window_tokenization(self, tokenizer, text, maxlen, stride):
        assert maxlen <= 510
        assert stride > 0
        all_tokens_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if hasattr(tokenizer, "cls_token"):
            CLS_ID = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
            SEP_ID = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
            PAD_ID = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        else:
            CLS_ID = tokenizer.convert_tokens_to_ids("[CLS]")
            SEP_ID = tokenizer.convert_tokens_to_ids("[SEP]")
            PAD_ID = tokenizer.convert_tokens_to_ids("[PAD]")
        ids_matrix = []
        idx_list = []
        sidx, eidx = (0, maxlen)
        full_len = 0
        while sidx < len(all_tokens_ids):
            real_end = min(sidx + maxlen, len(all_tokens_ids))
            idx_list.append((sidx, real_end))
            ids_matrix.append([CLS_ID] + all_tokens_ids[sidx:eidx] + [SEP_ID])
            full_len += len(ids_matrix[-1]) - 2
            if len(ids_matrix[-1]) < maxlen + 2:
                ids_matrix[-1].extend([PAD_ID] * (maxlen + 2 - len(ids_matrix[-1])))
            sidx, eidx = (sidx + stride, sidx + stride + maxlen)
        return (ids_matrix, idx_list)

    def tokens2pieces(self, text_strings, tokenizer):
        result = {}
        global_idx = 0
        all_tokens = []
        for sent_i, sent in enumerate(text_strings):
            spans = [(m.start(), m.end()) for m in re.finditer("\\S+", sent)]
            enc = tokenizer(
                sent,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
            tokens = tokenizer.convert_ids_to_tokens(enc.input_ids)
            all_tokens.extend(tokens)
            for start, end in enc.offset_mapping:
                for w_i, (s, e) in enumerate(spans):
                    if start >= s and end <= e:
                        result.setdefault((sent_i, w_i), []).append(global_idx)
                        break
                global_idx += 1
        self.TOKENS_LIST = all_tokens
        if global_idx != len(self.TOKENS_LIST):
            logging.getLogger(__name__).warning(
                f"tokens2pieces: counted {global_idx} subwords, but rebuilt TOKENS_LIST has {len(self.TOKENS_LIST)}"
            )
        return result

    def node2posn(self, nodeID):
        result = []
        sent_idx, bidx, eidx = [int(elt) for elt in nodeID.split("_")]
        for jdx in range(bidx, eidx + 1):
            result.extend(self.tokens2sub[sent_idx, jdx])
        return result


def initialize_tokenizer(encoder_type):
    if encoder_type != 'BERT':
        raise ValueError("The paper configurations require BERT-base-cased.")
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-cased", use_fast=True, add_prefix_space=False
    )
    return (tokenizer, encoder_type)


class TDG_file_reader:

    def __init__(self, filename, encoder_type=None):
        self.tokenizer, self.encoder_type = initialize_tokenizer(encoder_type)
        self.edge_vocab = Vocabulary()
        self.dataset = list(self.read_temporal_dataset(filename))

    def read_temporal_dataset(self, filename, n_docs=None, root_token="[ROOT]"):
        istream = open(filename)
        reader_state = "Out"
        TEXT, EDGES, NODES, TAGS, DLINKS = ([], [], [], [], [])
        examples = []
        cnt = 0
        for line in istream:
            if line == "" or line.isspace():
                reader_state = "Out"
                if TEXT:
                    cnt += 1
                    examples.append(
                        Example(
                            self.tokenizer,
                            TEXT,
                            dataset=filename,
                            nodelist=NODES,
                            taglist=TAGS,
                            govlist=DLINKS,
                            edgelist=EDGES,
                            root_token=root_token,
                            doc_id=cnt,
                        )
                    )
                    self.edge_vocab.update(EDGES)
                    TEXT = []
                    EDGES = []
                    NODES = []
                    TAGS = []
                    DLINKS = []
            elif line.startswith("filename:"):
                reader_state = "Text"
            elif line.startswith("EDGE_LIST"):
                reader_state = "Edges"
            elif reader_state == "Text":
                TEXT.append(line)
            elif reader_state == "Edges":
                node_id, node_type, node_gov, relation = line.strip().split()
                NODES.append(node_id)
                TAGS.append(node_type)
                DLINKS.append(node_gov)
                EDGES.append(relation)
        if TEXT:
            cnt += 1
            examples.append(
                Example(
                    self.tokenizer,
                    TEXT,
                    dataset=filename,
                    nodelist=NODES,
                    taglist=TAGS,
                    govlist=DLINKS,
                    edgelist=EDGES,
                    root_token=root_token,
                    doc_id=cnt,
                )
            )
            self.edge_vocab.update(EDGES)
        istream.close()
        return examples
