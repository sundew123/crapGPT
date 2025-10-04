gpt.zip - finished model, unzip and run main.py
streamModel - code used to generate model, download raw data from link in description, relies on test
test - encoder, lost the source code in accident, finds shortest length encoding of one or many csv-formatted strings in file (argument 2) using csv-formatted list of tokens specified in another file (argument 3)
vocab.txt - csv-formatted list of tokens used by model


To represent non-utf-8 symbols in the token list, use the format "\x" followed by the decimal number representing your symbol's byte value. Decimal number must be 3 or less characters or be less than 256, whichever comes first. If "\x" is not followed by a number, the encoder and decoder with treat it as the string "\x".
