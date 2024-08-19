""" from https://github.com/keithito/tacotron """
from text import cleaners
from text import symbols as symbol_dict
# import pdb

# Mappings from symbol to numeric ID and vice versa:
#_symbol_to_id = {s: i for i, s in enumerate(symbols)}
#_id_to_symbol = {i: s for i, s in enumerate(symbols)}

def get_symbols(expanded = False, korean = False):
  if expanded:
    symbols = [symbol_dict._pad] + list(symbol_dict._punctuation) + list(symbol_dict._letters) + list(symbol_dict._kletters) + list(symbol_dict._letters_ipa)
  elif korean:
    symbols = [symbol_dict._pad] + list(symbol_dict._punctuation) + list(symbol_dict._kletters)
  else:
    symbols = [symbol_dict._pad] + list(symbol_dict._punctuation) + list(symbol_dict._letters) + list(symbol_dict._letters_ipa)
  return symbols

# def _symbol_to_id(symbol, symbols):
#   return symbols.index(symbol)
def _symbol_to_id(symbol, symbols):
  try:
      return symbols.index(symbol)
  except ValueError:
      return None

def _id_to_symbol(id, symbols):
  return symbols[id]

def text_to_sequence(text, cleaner_names, symbols):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []
  clean_text = _clean_text(text, cleaner_names)
  # pdb.set_trace()
  # print(clean_text)
  # for symbol in clean_text:
  #   symbol_id = _symbol_to_id(symbol, symbols)
  #   sequence += [symbol_id]
  for symbol in clean_text:
    symbol_id = _symbol_to_id(symbol, symbols)
    if symbol_id is not None:  # If the symbol is in the symbols list
        sequence.append(symbol_id)
  return sequence

def cleaned_text_to_sequence(cleaned_text, symbols):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = [_symbol_to_id(symbol, symbols) for symbol in cleaned_text]
  return sequence


def sequence_to_text(sequence, symbols):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol(symbol_id, symbols)
    result += s
  return result


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text
