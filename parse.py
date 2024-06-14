import logging

import spacy

SPACY_MODEL_LG = spacy.load("ru_core_news_lg")
from pymystem3 import Mystem

MYSTEM = Mystem()


def get_all_names_spacy(text) -> list[tuple[int, int]]:
    doc = SPACY_MODEL_LG(text)
    names_positions = []
    for token in doc:
        print(
            f"{token.text} {token.pos_} {token.dep_} {token.head.text} {token.ent_type_}"
        )
        if token.ent_type_ == "PER":
            names_positions.append((token.idx, token.idx + len(token.text) - 1))
    return names_positions


def get_all_addresses_spacy(text) -> list[tuple[int, int]]:
    doc = SPACY_MODEL_LG(text)
    addresses_positions = []
    for token in doc:
        print(
            f"{token.text} {token.pos_} {token.dep_} {token.head.text} {token.ent_type_}"
        )
        if token.ent_type_ == "LOC":
            addresses_positions.append((token.idx, token.idx + len(token.text) - 1))
    return addresses_positions


def get_all_addresses_natasha(text) -> list[tuple[int, int]]:
    from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, NewsNERTagger, Doc

    natasha_segmenter = Segmenter()
    natasha_morph_vocab = MorphVocab()
    natasha_news_embedding = NewsEmbedding()
    natasha_morph_tagger = NewsMorphTagger(natasha_news_embedding)
    natasha_syntax_parser = NewsSyntaxParser(natasha_news_embedding)
    natasha_ner_tagger = NewsNERTagger(natasha_news_embedding)

    doc = Doc(text)
    doc.segment(natasha_segmenter)
    doc.tag_morph(natasha_morph_tagger)
    doc.parse_syntax(natasha_syntax_parser)
    doc.tag_ner(natasha_ner_tagger)

    logging.info(doc.spans)
    doc.ner.print()

    addresses_positions = []
    for span in doc.spans:
        if span.type == "LOC":
            addresses_positions.append((span.start, span.stop))
    return addresses_positions

def get_all_names_mystem(text) -> list[tuple[int, int]]:
    analyze = MYSTEM.analyze(text)
    names_positions = []
    last_index = 0
    previous_word_positions = []
    previous_word_tags = []
    # add positions if consecutive word tags are фам, имя, отч or фам, имя or имя отч фам or имя фам
    for word in analyze:
        try:
            analysis = word["analysis"][0]
        except (KeyError, IndexError):
            continue
        print(word)
        index = text.find(word["text"], last_index)

        if 'фам' in analysis['gr']:
            previous_word_tags.append("фам")
        elif 'имя' in analysis['gr']:
            previous_word_tags.append("имя")
        elif 'отч' in analysis['gr']:
            previous_word_tags.append("отч")
        else:
            previous_word_tags.append("")
        previous_word_positions.append((index, index + len(word["text"])))

        if previous_word_tags[-3:] in [["фам", "имя", "отч"], ["имя", "отч", "фам"]]:
            names_positions += previous_word_positions[-3:]
        elif previous_word_tags[-2:] in [["фам", "имя"], ["имя", "фам"]]:
            names_positions += previous_word_positions[-2:]

        last_index = index + len(word["text"])
    return names_positions
