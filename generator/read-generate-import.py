import argparse
import logging
from typing import List

import generator.input.file_operations
from generator.entities import WordWithContext, CardRawDataV1
from generator.input import read_input_file
from generator.anki import anki_importer, anki_operations
from generator.config import Config
from generator import generate_cards, entities
from generator import validation
from generator.generators import GeneratorConfig, GeneratorFactory, Generator, AvailableModels


def process_existing_cards(input_words: list[WordWithContext]) -> list[str]:
    filtered_words: list[WordWithContext] = validation.filter_words_are_present_in_deck(Config.DECK_NAME, input_words)
    input_words_without_context: list[str] = list(map(lambda word_with_context: word_with_context.word, filtered_words))

    logging.info("Processing existing cards")
    cards_in_directory: list[CardRawDataV1] = generator.input.file_operations.cards_in_directory(Config.PROCESSING_DIRECTORY_PATH)
    relevant_cards_in_directory: list[CardRawDataV1] = [card for card in cards_in_directory if card.word in input_words_without_context]
    existing_cards_validated: list[CardRawDataV1] = validation.discard_invalid_cards(Config.PROCESSING_DIRECTORY_PATH, relevant_cards_in_directory)
    existing_cards_data: dict[WordWithContext, CardRawDataV1] = entities.cards_to_dict(existing_cards_validated)
    anki_importer.import_card_collection(existing_cards_data)
    logging.info("All existing cards processed")

    imported_existing_words: list[str] = [item.word for item in existing_cards_data.keys()]
    return imported_existing_words


def process_new_cards(input_words: list[WordWithContext], generators: List[Generator]):
    filtered_words: list[WordWithContext] = validation.filter_words_are_present_in_deck(Config.DECK_NAME, input_words)
    generated_cards_data: dict[WordWithContext, CardRawDataV1] = generate_cards.generate_text_and_image(filtered_words, generators)
    logging.info("Card generation completed")
    anki_importer.import_card_collection(generated_cards_data)
    logging.info("Import in Anki completed")


def exclude_imported_words(input_words, imported_existing_words) -> list[WordWithContext]:
    if len(imported_existing_words) >= 1:
        logging.info(f"Words {imported_existing_words} are imported from existing files and are excluded from further processing")
        return list(filter(lambda word_with_context: word_with_context.word not in imported_existing_words, input_words))
    return input_words


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="This tool processes the list of words or phrases (with optional context) and creates an Anki card for each word. The cards are then imported via AnkiConnect.")

    # Required positional arguments
    parser.add_argument('input_file', type=str, help="Path to the input file, CSV with semicolons or Excel. Header [word;context] is required")
    parser.add_argument('processing_directory', type=str, help="Path of the directory where the data should be processed. It could be an empty directory. If directory contains generated cards, tool will suggest to import them.")

    # Optional arguments
    parser.add_argument('--openai_api_key', type=str, help="API key for OpenAI. If not set, the value from environment variable OPENAI_API_KEY is used",
                        default="")
    parser.add_argument('--deck_name', type=str, help="Name of the Anki deck. If not set, the default name is generated", default=None)
    parser.add_argument(
        '--anki_media_directory_path',
        type=str,
        help="Path to the Anki media directory. If not set, the standard path for each OS is used",
        default="/Users/photosartd/Library/Application Support/Anki2/Benutzer 1/collection.media")
    parser.add_argument('--language', type=str, help="Target card language. Not only the card translation, customized generation process for each language", default=Config.DEFAULT_LANGUAGE, choices=Config.SUPPORTED_LANGUAGES)
    parser.add_argument("--level", type=str, help="Current language level, that should be used for card creation to avoid overcomplicated cards for beginners and vice versa", default=Config.DEFAULT_LEVEL, choices=Config.SUPPORTED_LEVELS)

    #Api Keys and Models
    parser.add_argument("--text_model", type=str, default=AvailableModels.COHERE.value)
    parser.add_argument("--text_api_key", type=str, help="Text model API key",
                        default="alLJoMiQJmwGYhoSDNUn5nPrIPKOZWxAKcTIxgfb")
    parser.add_argument("--text_secret_key", type=str, help="Text model secret key", default="")

    parser.add_argument("--image_model", type=str, default=AvailableModels.KANDINSKY_3.value)
    parser.add_argument("--image_api_key", type=str, help="Image model API key",
                        default="D4A1B409BF9DB0CB3BA89D5507A2C7E8")
    parser.add_argument("--image_secret_key", type=str, help="Image model secret",
                        default="E6A4DAF80A44A4935EE4A7DC9C15C294")

    parser.add_argument("--voice_model", type=str, default=AvailableModels.ELEVENLABS.value)
    parser.add_argument("--voice_api_key", type=str, help="Voice model API",
                        default="08d45505b4cf43add13b1e02128bfc41")
    parser.add_argument("--voice_secret_key", type=str, help="Voice model secret", default="")

    # Parse arguments
    args = parser.parse_args()

    # Setup config
    Config.setup_logging()
    Config.set_openai_key_or_use_default(args.openai_api_key)
    Config.set_anki_deck_name_or_use_default(args.deck_name)
    Config.set_anki_media_directory_or_use_default(args.anki_media_directory_path)
    Config.set_processing_directory_path(args.processing_directory)
    Config.set_language_or_use_default(args.language)
    Config.set_level_or_use_default(args.level)

    # validate environment and read inputs
    validation.check_anki_connect()
    validation.check_whether_deck_exists()
    validation.check_language()
    input_words: list[WordWithContext] = read_input_file.read_file_based_on_extension(args.input_file)

    # Processing
    imported_existing_words: list[str] = process_existing_cards(input_words)
    logging.info("Existing cards processed")
    input_words_except_imported = exclude_imported_words(input_words, imported_existing_words)
    generators_configs: List[GeneratorConfig] = GeneratorConfig.from_args(args)
    generators: List[Generator] = list(
        map(
            lambda generator_config: GeneratorFactory.create(generator_config),
            generators_configs
        )
    )
    process_new_cards(input_words_except_imported, generators)
    logging.info("New cards processed")
    logging.info("Processing completed")


if __name__ == '__main__':
    main()
