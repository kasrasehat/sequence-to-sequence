from essential_func import readLangs, prepareData
import random


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))
