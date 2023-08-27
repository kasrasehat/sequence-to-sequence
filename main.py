from essential_func import prepareData, get_dataloader, get_dataloader, train, evaluateRandomly, evaluateAndShowAttention
import random
import torch
from Networks import EncoderRNN, DecoderRNN, AttnDecoderRNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
hidden_size = 128
batch_size = 32
input_lang, output_lang, train_dataloader = get_dataloader(batch_size)

encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

train(train_dataloader, encoder, decoder, 100, print_every=5, plot_every=5)
encoder.eval()
decoder.eval()
evaluateRandomly(encoder=encoder, decoder=decoder, n=10)
evaluateAndShowAttention('il n est pas aussi grand que son pere', encoder=encoder, decoder=decoder)

evaluateAndShowAttention('je suis trop fatigue pour conduire', encoder=encoder, decoder=decoder)

evaluateAndShowAttention('je suis desole si c est une question idiote', encoder=encoder, decoder=decoder)

evaluateAndShowAttention('je suis reellement fiere de vous', encoder=encoder, decoder=decoder)


