import string
import streamlit as st
import stanza
import torch
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM


@st.cache
def download_bert_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-cased")


@st.cache
def download_bert_model():
    return BertForMaskedLM.from_pretrained("bert-base-cased")


def unique_tags_collect(mask_words_list):
    unique_tag = {tag for word, tag in mask_words_list}
    return list(unique_tag)


def random_picker(tag_list):
    list_len = len(tag_list)
    index = np.random.randint(list_len)
    return tag_list[index]


def format_question2(real_question, four_options, ans, ABCD):
    final_options = [i + option + "  " for i, option in zip(ABCD, four_options)]
    answer = [
        option[:-1] for option in final_options if option.endswith(ans[0] + "  ")
    ][0]
    final_options.insert(0, real_question + " ")
    return final_options, answer


def cloze_test(text, nlp, pos_tags, tokenizer, model, ABCD):
    doc = nlp(text)
    mask_words = [
        (word.text, word.upos)
        for word in doc.sentences[0].words
        if word.upos in pos_tags
    ]

    unique_tags = unique_tags_collect(mask_words)
    random_tag = random_picker(unique_tags)
    mask_words_with_same_tag = [
        (word.text, word.upos)
        for word in doc.sentences[0].words
        if word.upos == random_tag
    ]
    ans = random_picker(mask_words_with_same_tag)

    raw_question = doc.sentences[0].text
    masked_question = raw_question.replace(ans[0], "[MASK]")
    real_question = masked_question.replace("[MASK]", "______")

    input_ids = tokenizer.encode(masked_question, add_special_tokens=True)
    masked_index = input_ids.index(103)  # '[MASK]': 103

    outputs = model(torch.tensor(input_ids)[None, :])  # Batch size 1
    prediction_scores = outputs[0]

    _, indices = torch.topk(prediction_scores[0, masked_index], k=100)
    options = tokenizer.convert_ids_to_tokens(indices)[-50:]
    options = [
        option
        for option in options
        if option != ans[0]
        and not option.startswith("##")
        and option not in string.punctuation
    ]
    three_options = np.random.choice(options, 3, replace=False)
    four_options = np.append(three_options, ans[0])
    four_options = np.random.permutation(four_options)

    question, answer = format_question2(real_question, four_options, ans, ABCD)

    return question, answer


def generate(nlp, text, tokenizer, model):
    pos_tags = ["VERB", "ADJ", "ADV", "NOUN"]
    ABCD = ["(A) ", "(B) ", "(C) ", "(D) "]
    doc = nlp(text)
    questions = []
    answers = []
    options = []

    for sentence in doc.sentences:
        question, answer = cloze_test(
            sentence.text, nlp, pos_tags, tokenizer, model, ABCD
        )
        questions.append(question)
        answers.append(answer)

    paragraph = r""
    for i, question in enumerate(questions, start=1):
        paragraph += question[0].replace("______", f"__({i})__")

    for i, question in enumerate(questions, start=1):
        option_text = ""
        for option in question[1:]:
            option_text += f"{option:20s}"
        option_text = "______ " + f"{i}. " + option_text
        options.append(option_text.rstrip())
    return questions, answers, options, paragraph


def main():
    st.set_page_config(page_title="Cloze Test Generation", page_icon="🦈", layout="wide")

    st.markdown(
        "<h1 style='text-align: center;'> Cloze Test Generation</h1>",
        unsafe_allow_html=True,
    )

    st.header("Paste, Generate and Test!")

    user_name = st.text_input("Enter your name: ")
    if user_name:
        st.info(
            f"""Hi, {user_name}. This application is based on the following libraries.
            \n - [Transformers](https://github.com/huggingface/transformers)
            \n - [Stanza](https://github.com/stanfordnlp/stanza)
            \n - [PyTorch](https://pytorch.org/)
            \n - [NumPy](https://numpy.org/)
            """
        )

        st.write("""### We'll prepare the toolkits before we start.""")
        st.write("""This process could take a while for the first time.""")
        st.warning("Preparing Stanza toolkits...")
        try:
            nlp = stanza.Pipeline(
                lang="en", processors="tokenize,pos", verbose=0, use_gpu=False
            )
        except:
            stanza.download("en")
            nlp = stanza.Pipeline(
                lang="en", processors="tokenize,pos", verbose=0, use_gpu=False
            )

        st.warning("Preparing BERT tokenzier...")
        tokenizer = download_bert_tokenizer()

        st.warning("Preparing BERT model...")
        model = download_bert_model()

        st.success("Done")

        with st.form(key="text_form"):
            st.write("""### Paste your article below. We'll do the rest for you!""")
            text = st.text_area(
                "Here is a default text. Change it to whatever you want and press Generate!",
                """In the 1980s the number of giant pandas in China hovered around 1,100. Now, after decades of focused conservation, giant pandas have been crossed off the endangered list. Habitat preservation, anti-poaching efforts, and advances in captive-breeding programs can offer a lifeline to the most endangered members of the biosphere.""",
            )

            generate_btn = st.form_submit_button(label="Generate")
            if generate_btn:
                questions, answers, options, paragraph = generate(
                    nlp, text, tokenizer, model
                )

                st.text_area("Result: ", paragraph)

                for option in options:
                    st.text(option)

                with st.beta_expander("Answer Section"):
                    for i, answer in enumerate(answers, start=1):
                        st.text(f"{i}. " + answer)


if __name__ == "__main__":
    main()
