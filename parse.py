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


import re

def get_address_indices_algo(text):
    address_levels = [
        ["респ\.","республика","край","область","обл\.","г\.ф\.з\.","а\.окр\."],
        ["пос\.","поселение","р-н","район","с\/с","сельсовет"],
        ["г\.","город","пгт\.","рп\.","кп\.","гп\.","п\.","поселок","аал","арбан","аул","в-ки","выселки","г-к","заимка","з-ка","починок","п-к","киш\.","кишлак","п\.ст\.","ж\/д","м-ко","местечко","деревня","с\.","село","сл\.","ст\.","станция","ст-ца","станица","у\.","улус","х\.","хутор","рзд\.","разъезд","зим\.","зимовье","д\."],
        ["ал\.","аллея","б-р","бульвар","взв\.","взд\.","въезд","дор\.","дорога","ззд\.","заезд","километр","к-цо","кольцо","лн\.","линия","мгстр\.","магистраль","наб\.","набережная","пер-д","переезд","пер\.","переулок","пл-ка","площадка","пл\.","площадь","пр-кт\.","проспект","проул\.","проулок","рзд\.","разъезд","ряд","с-р","сквер","с-к","спуск","сзд\.","съезд","тракт","туп\.","тупик","ул\.","улица","ш\.","шоссе"],
        ["влд\.","владение","г-ж","гараж","д\.","дом","двлд\.","домовладение","зд\.","здание","з\/у","участок","кв\.","квартира","ком\.","комната","подв\.","подвал","кот\.","котельная","п-б","погреб","к\.","корпус","офис","пав\.","павильон","помещ\.","помещение","раб\.уч\.","скл\.","склад","соор\.","сооружение","стр\.","строение","торг\.зал\.","цех"]
    ]

    address_regex = re.compile(
        r'(' + '|'.join([r'|'.join(level) for level in address_levels]) + r')',
        re.IGNORECASE
    )

    matches = []
    for match in address_regex.finditer(text):
        matches.append((match.start(), match.end()))

    return matches

def find_addresses_algo(text):
    indices = get_address_indices_algo(text)
    current_address = ""
    start_index = -1
    end_index = -1
    addresses = []

    for start, end in indices:
        if start_index == -1:
            start_index = start

        part = text[start:end]
        current_address += part + " "
        end_index = end

        if re.search(r'[,.!?;]', text[end:]):
            addresses.append((current_address.strip(), start_index, end_index))
            current_address = ""
            start_index = -1
            end_index = -1

    if current_address:
        addresses.append((current_address.strip(), start_index, end_index))

    return [(start, end) for _, start, end in addresses]


def get_phone_numbers_positions(text) -> list[tuple[int, int]]:
    matches = re.findall(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]", text)
    positions = []
    for match in matches:
        start = text.find(match)
        end = start + len(match) - 1
        positions.append((start, end))
    return positions


def get_bd_positions(text) -> list[tuple[int, int]]:
    bd_positions = []
    for match in re.finditer(r"\d{2}\.\d{2}\.\d{4}", text):
        bd_positions.append((match.start(), match.end()))
    return bd_positions
