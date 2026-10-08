"""Static word2vec vectors for the dictionary-embedding robustness check.

English: GoogleNews-vectors-negative300 (Twitter, surveys); Chinese: SGNS
merge (Weibo). Model paths come from config (WORD2VEC_ENGLISH /
WORD2VEC_CHINESE). Used by analysis.semantic.semantic_similarity with
``--embedding_type dictionary``; run_word2vec.sh drives the full check.
"""

from gensim.models import KeyedVectors

from config import cfg


class ChineseDictionaryEmbedding:
    def __init__(self, model_path=None):
        model_path = model_path or cfg.WORD2VEC_CHINESE
        """
        model_path: Chinese word2vec file (text format), e.g. sgns.merge.word
        """
        print(f"Loading Chinese word2vec model from: {model_path}")

        # 使用 KeyedVectors.load_word2vec_format 加载 SGNS 模型
        # 注意：sgns.merge.word 是 word2vec 的 text格式，而不是 gensim .model
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=False)

        print("Model loaded! Total vocab words:", len(self.model))

        print(self.model.index_to_key[:10])

    def get_embedding(self, keyword):
        """
        返回一个 numpy 数组（shape=(300,)）
        如果词不存在，返回 None
        """
        if keyword in self.model:
            return self.model[keyword]
        return None

    def output_vocabs(self, vocabs_file):
        with open(vocabs_file, "w") as f:
            # 过滤掉 None 值，只保留有效的词汇
            valid_vocabs = [
                vocab for vocab in self.model.index_to_key if vocab is not None
            ]
            f.write("\n".join(valid_vocabs))


class EnglishDictionaryEmbedding:
    def __init__(
        self,
        model_path=None,
    ):
        model_path = model_path or cfg.WORD2VEC_ENGLISH
        print("Loading model...")
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        print("Model loaded!")

    def get_embedding(self, keyword):
        """
        返回单词的300维向量，如果词不存在返回 None
        """
        if keyword in self.model:
            return self.model[keyword]
        return None

    def output_vocabs(self, vocabs_file):
        with open(vocabs_file, "w") as f:
            # 过滤掉 None 值，只保留有效的词汇
            valid_vocabs = [
                vocab for vocab in self.model.index_to_key if vocab is not None
            ]
            f.write("\n".join(valid_vocabs))


if __name__ == "__main__":
    # eng_embed = EnglishDictionaryEmbedding()
    # eng_embed.get_embedding("abortion")
    # eng_embed.get_embedding("gun")
    # eng_embed.output_vocabs("eng_vocabs.txt")

    chi_embed = ChineseDictionaryEmbedding()
    chi_embed.get_embedding("美国")
    chi_embed.get_embedding("俄罗斯")
    chi_embed.output_vocabs("chi_vocabs.txt")
