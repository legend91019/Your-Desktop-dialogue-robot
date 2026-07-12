import torch
import random
import jieba


class DataAugmenter:
    """文本数据增强器"""

    def __init__(self):
        self.synonyms = {
            "怎么": ["如何", "怎样", "咋"],
            "天气": ["气候", "气象"],
            "安装": ["设置", "配置", "装载"],
            "下雨": ["降雨", "落雨"],
            "分析": ["解析", "剖析", "研究"],
            "最近": ["近期", "近来", "近日"],
            "爆发": ["发生", "引发", "产生"],
            "谁": ["何人", "什么人"],
            "提出": ["发明", "创立", "发现"],
        }

    def synonym_replacement(self, text, n=1):
        words = list(jieba.cut(text))
        new_words = words.copy()
        replaceable_words = [w for w in words if w in self.synonyms]

        if not replaceable_words:
            return text

        random.shuffle(replaceable_words)
        for i in range(min(n, len(replaceable_words))):
            word = replaceable_words[i]
            new_words = [
                random.choice(self.synonyms[word]) if w == word else w
                for w in new_words
            ]
        return "".join(new_words)

    def random_insertion(self, text, n=1):
        words = list(jieba.cut(text))
        if len(words) < 2:
            return text

        for _ in range(n):
            word = random.choice(words)
            if word in self.synonyms:
                synonym = random.choice(self.synonyms[word])
                insert_pos = random.randint(0, len(words))
                words.insert(insert_pos, synonym)
        return "".join(words)

    def random_swap(self, text, n=1):
        words = list(jieba.cut(text))
        if len(words) < 2:
            return text

        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return "".join(words)

    def random_deletion(self, text, p=0.2):
        words = [w for w in jieba.cut(text) if random.random() > p]
        return "".join(words) if words else text[:1]


def prepare_data(tokenizer, questions, labels, augmenter=None, augment_times=3):
    if augmenter and augment_times > 0:
        augmented_data = []
        for q, lbl in zip(questions, labels):
            augmented_data.append((q, lbl))
            for _ in range(augment_times):
                method = random.choice([
                    augmenter.synonym_replacement,
                    augmenter.random_insertion,
                    augmenter.random_swap,
                    augmenter.random_deletion,
                ])
                new_q = method(q)
                if new_q != q:
                    augmented_data.append((new_q, lbl))

        texts, lbls = zip(*augmented_data)
    else:
        texts, lbls = questions, labels

    inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
    return inputs, torch.tensor(lbls)
