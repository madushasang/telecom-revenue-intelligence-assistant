from transformers import AutoTokenizer

MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    texts = [
        "Prepaid recharge revenue decreased unexpectedly.",
        "MSISDN",
        "IMSI",
        "CDR",
        "interconnect",
        "revenue assurance",
        "TAP3",
        "VoLTE",
    ]

    for text in texts:
        tokens=tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text,add_special_tokens=False)
        token_ids_with_special_tokens  = tokenizer.encode(text,add_special_tokens=True)

        print("=" *60) 
        print(f"TEXT :{text}")
        print(f"TOKENS : {tokens}")
        print(f"TOEKN IDS : {token_ids}")
        print(f"TOEKN IDS WITH SPCL : {token_ids_with_special_tokens}")
        print("WITH TOEKNS:",tokenizer.convert_ids_to_tokens(token_ids_with_special_tokens))
        print(f"COUNT : {len(token_ids)}")

if __name__=="__main__":
    main()