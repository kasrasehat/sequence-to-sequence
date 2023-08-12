from essential_func import prepareData, get_dataloader
import random


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))
