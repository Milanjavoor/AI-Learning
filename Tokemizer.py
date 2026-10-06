import re
import json 
from pathlib import Path
from collections import Counter
from typing import List,Tuple,Dict,Optional

# Thsi is a word level tokenizer . 

class Tokenizer:
    # special tokens
    pad_tok="<pad>"
    sos_tok="<sos>"# start of series token
    eos_tok="<eos>"
    unknown_tok="<ukn>"

    def __init__(self,vocab_size:int=10000):
        self.vocab_size=vocab_size
        self.words_to_ids:Dict[str,int]={}
        self.ids_to_words:Dict[int,str]={}
        self.word_freq:Counter=Counter()
        self.is_built=False

    @staticmethod
    def normalise_text(text):
    #Normalizes text , lowerizes it , removes extra whitespace and basic punctuation handling
        text=text.lower()
        #Removing extra whitespace
        text=re.sub(r'\s+',' ',text).strip()
        #Seperating punchuation from words . Dependent on language
        text=re.sub(r'(?<=[a-z])([,!?.;:\(\)])', r' \1',text)
        text=re.sub(r'([,!?.;:\(\)])(?=[a-z])', r'\1 ',text)

        return text
    @staticmethod
    def Tokenize(text):
    # Splits the text into tokens . Word tokenizer
        return text.split()

    def build_vocabulary(self,texts,min_freq=2):
        # Build vocabulary from list of texts 
        print("Buliding vocabulary from ",len(texts),"texts")
        for text in texts:
            normalize=self.normalise_text(text)
            tokens=self.Tokenize(normalize)
            self.word_freq.update(tokens)
         # Reserve space for four special tokens
        vocab_size_for_words=self.vocab_size-4

        common_words=[
            word for word,frequency in self.word_freq.most_common(vocab_size_for_words)
            if frequency>=min_freq
        ]

        self.words_to_ids={
            self.pad_tok:0,
            self.sos_tok:1,
            self.eos_tok:2,
            self.unknown_tok:3}

        
        # Build vocabulary for the dictionary word to indexes
        for idx,word in enumerate(common_words,start=4):
            self.words_to_ids[word]=idx

        #  build the reverse dictionary
        self.ids_to_words={idx:word for word,idx in self.words_to_ids.items()}
       
        self.is_built=True


        print("Vocabulary built with",len(self.words_to_ids),"tokens")
        print("Number of words in the vocabulary",len(common_words))
        print(f"Coverage:,{len(common_words)/vocab_size_for_words*100:.1f}")

    def encode(self,text,max_len,add_special_tokens=True):
        # Encode text tokens into token ids

        if not self.is_built:
            raise ValueError("Vocabulary must be built before encoding")
        normalized=self.normalise_text(text)
        toekns=self.Tokenize(normalized)

        # Convert into token ids
        token_ids=[]
        # Add starting token 
        if add_special_tokens:
            token_ids.append(self.words_to_ids[self.sos_tok])
        # Add each token id if not available use unknown token
        for toekn in toekns:
            tokenid=self.words_to_ids.get(toekn,self.words_to_ids[self.unknown_tok])
            token_ids.append(tokenid)
        # Add end of series token
        if add_special_tokens:
            tokenid.append(self.words_to_ids[self.eos_tok])

    # If required truncate the size
        if max_len is not None and len(token_ids>max_len):
            token_ids=token_ids[:max_len-1]
            if add_special_tokens:
                token_ids.append(self.words_to_ids[self.eos_tok])
        return token_ids
        
    def decode(self,tokenids,remove_special_tokens):

    # Convert tokenids back into regular texts and removes special tokens
        tokens=[]
        for id in tokenids:
            if id not in self.ids_to_words:
                continue

            token=self.ids_to_words[id]

            if remove_special_tokens and id in [self.sos_tok,self.eos_tok,self.unknown_tok]:
                continue

            tokens.append(token)
# Joining all the tokens with space between them to create a text sequence
        text=" ".join(tokens)
# Basic detokenization 
        text = re.sub(r'\s+([,!?.;:\)\]"])', r'\1', text)
        text = re.sub(r'([\[\("])\s+', r'\1', text)

        return text

    def return_vocab_size(self):
        return len(self.words_to_ids)

    
    def get_pad_token_id(self):
        return self.words_to_ids[self.pad_tok]

    
    def get_start_token_id(self):
        return self.words_to_ids[self.sos_tok]

    def get_unknown_token_id(self):
        return self.words_to_ids[self.unknown_tok]

    def get_end_token_id(self):
        return self.words_to_ids[self.eos_tok]


    def save_vocabulary(self,filepath):
        data={
            "word_to_id":self.words_to_ids,
            "word_freq":dict(self.word_freq.most_common(1000)),
            "id_to_word":{str(k):v for k,v in self.ids_to_words.items()}
        }
        filepath.parent.mkdir(parents=True,exists_ok=True)
        with open (filepath,"w",encoding="utf-8") as f:
            json.dump(data,filepath,ensure_ascii=False,)
            
            token_ids=token_ids[:max_len-1]
            if add_special_tokens:
                token_ids.append(self.words_to_ids[self.eos_tok])
        return token_ids
