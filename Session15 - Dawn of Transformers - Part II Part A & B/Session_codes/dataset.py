import torch
import torch.nn
from torch.utils.data import Dataset

## convert from one language to another
class BillingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype = torch.int64)

    def __len__(self):
        return len(self.ds)


    def __getitem__(self, idx):
        ## extracting the text fromt he input
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        ## transform text into token
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        ##add sos eos and padding to each of the sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 ## add both sos and eod
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 ## add only sos and not eos

        ## make sure number of padding tokenn is not negative. If it is, sentence is too long
        if enc_num_padding_tokens < 0  or dec_num_padding_tokens < 0:
            raise ValueError("Sentence too long")

        ## add sos and eos token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens , dtype = torch.int64),
            ],
            dim = 0,
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens , dtype = torch.int64),
            ],
            dim = 0,
        )

        ## add only eos token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens , dtype = torch.int64),
            ],
            dim = 0,
        )

        ## NOTICE THE DIFFERENCE b/w DECODER_INPUT and LABEL, this difference allows us to parallely train decoder models
        ## for any index i, input is from 0 to i of decoder input and label(or prediction) is ith of label which is actually the next word.

        # double check the size of tensors to make sure they are fo same length i.e. seq_len
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input" : encoder_input,
            "decoder_input" : decoder_input,
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  ## (1,1,seq_len)
            # where ever encoder token is not equal to pad token, pass TRUE, and where it is equal to pad pass FALSE , thereforE of type(T, T ,T, F, F, F, F)
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1,seq_len) & (1, seq_len, seq_len)
            ## seq_len = 10
            ## SOS    I  GOT   A   CAT    PAD    PAD    PAD    PAD    PAD    PAD
            ## TRUE TRUE TRUE TRUE TRUE  FALSE  FALSE  FALSE  FALSE  FALSE  FALSE
            ## 1 1 1 1 1 0 0 0 0 0
            ## Upper triangular matrix
            ## 1 1 1 1 1 1 1 1 1 1
            ## 0 1 1 1 1 1 1 1 1 1
            ## 0 0 1 1 1 1 1 1 1 1
            ## 0 0 0 1 1 1 1 1 1 1
            ## 0 0 0 0 1 1 1 1 1 1
            ## 0 0 0 0 0 1 1 1 1 1
            ## 0 0 0 0 0 0 1 1 1 1
            ## 0 0 0 0 0 0 0 1 1 1
            ## 0 0 0 0 0 0 0 0 1 1
            ## 0 0 0 0 0 0 0 0 0 1

            ## after AND operation - Final Decoder Mask
            ## 1 1 1 1 1 0 0 0 0 0
            ## 0 1 1 1 1 0 0 0 0 0
            ## 0 0 1 1 1 0 0 0 0 0
            ## 0 0 0 1 1 0 0 0 0 0
            ## 0 0 0 0 1 0 0 0 0 0
            ## 0 0 0 0 0 0 0 0 0 0
            ## 0 0 0 0 0 0 0 0 0 0
            ## 0 0 0 0 0 0 0 0 0 0
            ## 0 0 0 0 0 0 0 0 0 0
            ## 0 0 0 0 0 0 0 0 0 0
            "label" : label, #(seq_len)
            "src_text" : src_text,
            "tgt_text" : tgt_text,
        }

def causal_mask(size):
    ## creates upper traigular matrix of ones with diagonal = 1.
    mask = torch.triu(torch.ones((1, size, size)), diagonal = 1).type(torch.int)
    return mask == 0
