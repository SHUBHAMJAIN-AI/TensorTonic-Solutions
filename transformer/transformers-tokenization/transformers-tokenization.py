import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        uniqueset= set()
        self.id_to_word={0:"<PAD>", 1:"<UNK>",2:"<BOS>",3:"<EOS>"}
        self.word_to_id = {"<PAD>":0,"<UNK>":1,"<BOS>":2,"<EOS>":3}
        for s in texts :
            for word in s.split():
                if word not in uniqueset :
                    uniqueset.add(word)
        for i,j in enumerate(sorted(list(uniqueset))):
            self.id_to_word[i+4]=j
            self.word_to_id[j]=i+4
        self.vocab_size = len(self.word_to_id)
        return None
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        output =[]
        for s in text.split():
            output.append(self.word_to_id.get(s,1))
        return output
            
        
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        doutput=[]
        for s in ids :
            doutput.append(self.id_to_word.get(s,self.unk_token))
        
        return " ".join(doutput)
