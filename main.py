from essential_func import prepareData, get_dataloader, get_dataloader, train, evaluateRandomly
import random
import torch
from Networks import EncoderRNN, DecoderRNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
hidden_size = 128
batch_size = 32
input_lang, output_lang, train_dataloader = get_dataloader(batch_size)

encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = DecoderRNN(hidden_size, output_lang.n_words).to(device)

train(train_dataloader, encoder, decoder, 100, print_every=5, plot_every=5)
encoder.eval()
decoder.eval()
evaluateRandomly(encoder=encoder, decoder=decoder, n=10)



