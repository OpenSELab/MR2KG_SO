from transformers import T5Tokenizer, T5ForConditionalGeneration
import utils
import os

tokenizer = T5Tokenizer.from_pretrained(r"D:\CYW\roberta\summary\model")
model = T5ForConditionalGeneration.from_pretrained(r"D:\CYW\roberta\summary\model")


def dataSummary(nodelist):
    for node in nodelist:
        t_body = node["body"]
        ARTICLE_TO_SUMMARIZE = t_body
        if node["summaryBody"] == "":
            # generate summary
            input_ids = tokenizer.encode(ARTICLE_TO_SUMMARIZE, return_tensors='pt')
            summary_ids = model.generate(input_ids,
                        min_length=20,
                        max_length=80,
                        num_beams=10,
                        repetition_penalty=2.5,
                        length_penalty=1.0,
                        early_stopping=True,
                        no_repeat_ngram_size=2,
                        use_cache=True,
                        do_sample = True,
                        temperature = 0.8,
                        top_k = 50,
                        top_p = 0.95)

            summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            node["summaryBody"] = summary_text
    return nodelist
