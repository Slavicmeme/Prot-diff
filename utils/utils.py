import torch
from transformers import T5Tokenizer, T5EncoderModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
encoder = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").eval().to(device)

def encode_sequence(seq):
    tokens = tokenizer(' '.join(seq), return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = encoder(**tokens)
    return outputs.last_hidden_state.mean(1).squeeze(0)
