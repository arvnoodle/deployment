import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import spacy
from spacy import displacy


nlp = spacy.load("en_core_web_sm")

import neattext as nt
import neattext.functions as nfx
from textblob import TextBlob
from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def text_analyzer(my_text):
    docx = nlp(my_text)
    allData = [(token.text, token.shape, token.pos_,
                token.tag_, token.lemma_, token.is_alpha,
                token.is_stop) for token in docx]

    df = pd.DataFrame(allData, columns=['Token', 'Shape', 'PoS', 'Tag', 'Lemma',
                                        'IsAlpha', 'Is_Stopword'])
    return df


def get_entities(my_text):
    docx = nlp(my_text)
    entities = [(entity.text, entity.label_) for entity in docx.ents]
    return entities


HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""


# @st.cache
def render_entities(rawtext):
    docx = nlp(rawtext)
    html = displacy.render(docx, style="ent")
    html = html.replace("\n\n", "\n")
    result = HTML_WRAPPER.format(html)
    return result

def get_most_common_tokens(my_text,num=4):
    word_tokens = Counter(my_text.split())
    most_common_tokens = word_tokens.most_common(num)
    return most_common_tokens

def get_sentiment(my_text):
    blob = TextBlob(my_text)
    sentiment = blob.sentiment
    return sentiment

def main():
    st.title("NLP App")
    menu = ["Home", "NLP(files)", "About"]

    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Home: Analyze Text")
        raw_text = st.text_area("Enter Text Here")
        num_of_most_common = st.sidebar.number_input("Most Common Tokens", 5, 15)
        if st.button("Analyze"):
            with st.expander("Original Text"):
                st.write(raw_text)

            with st.expander("Text Analysis"):
                token_result_df = text_analyzer(raw_text)
                st.dataframe(token_result_df)

            with st.expander("Entities"):
                # entity_result = get_entities(raw_text)
                # st.write(entity_result)

                entity_result = render_entities(raw_text)
                stc.html(entity_result, height=1000,scrolling=True)


            col1, col2 = st.columns(2)

            with col1:
                with st.expander("Word Stats"):
                    st.info("Word Statistics")
                    docx = nt.TextFrame(raw_text)
                    st.write(docx.word_stats())

                with st.expander("Top Keywords"):
                    st.info("Top Keywords/Tokens")
                    processed_text = nfx.remove_stopwords(raw_text)
                    keywords = get_most_common_tokens(processed_text)
                    st.write(keywords)

                with st.expander("Sentiment"):
                    sent_result = get_sentiment(raw_text)
                    st.write(sent_result)

            with col2:
                with st.expander("Plot Word Freq"):
                    fig = plt.figure()
                    # sns.countplot(token_result_df['Token'])
                    # plt.xticks(rotation=90)
                    top_keywords = get_most_common_tokens(processed_text)
                    plt.bar(dict(keywords).keys(), dict(top_keywords).values())
                    st.pyplot(fig)

                with st.expander("Plot Part of Speech"):
                    fig = plt.figure()
                    sns.countplot(token_result_df['PoS'])
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                with st.expander("Plot Word Cloud"):
                    pass

            with st.expander("Download Text Analysis Results"):
                pass






    elif choice == "NLP(files)":
        st.subheader("NLP Task")

    else:
        st.subheader("About")


if __name__ == "__main__":
    main()
