def sentence_cleaner(src_type, sentence: str):
    import re

    if src_type == "tweet":
        sentence = sentence.replace('"', "")
        sentence = sentence.replace("RT", "")
        sentence = sentence.replace(".", "")
        sentence = sentence.replace("'", "")
        results = re.compile(r"[http|https]*://[a-zA-Z0-9.?/&=:_%,-~]*", re.S)
        sentence = re.sub(results, "", sentence)
        sentence = re.sub("[\u4e00-\u9fa5]", "", sentence)
        # results2 = re.compile(r'[@].*?[ ]', re.S)
        # sentence = re.sub(results2, '', sentence)
        sentence = sentence.replace("\n", " ")
        sentence = sentence.strip()
        results2 = re.compile(r"[@].*?[ ]", re.S)
        sentence = re.sub(results2, "", sentence)
        return sentence
    if src_type == "weibo":
        sentence = sentence.replace("“", "")
        sentence = sentence.replace("”", "")
        sentence = sentence.replace("…", "")
        sentence = sentence.replace("点击链接查看更多->", "")
        results = re.compile(
            r"[a-zA-Z0-9.?/&=:_%,-~#《》]", re.S
        )  # 。，：；“”‘’【】（） ]', re.S)
        # results = re.compile(r'[http|https]*://[a-zA-Z0-9.?/&=:_%,-~]*', re.S)
        sentence = re.sub(results, "", sentence)
        results2 = re.compile(r"[//@].*?[:]", re.S)
        sentence = re.sub(results2, "", sentence)
        sentence = sentence.replace("\n", " ")
        sentence = sentence.strip()
        return sentence
    return sentence


from torch.utils.data import Dataset


class OpinionDataset(Dataset):
    def __init__(self, all_dataset) -> None:
        super().__init__()
        self.all_dataset = all_dataset

    def __getitem__(self, index):
        example = self.all_dataset[index]
        input_ids, attention_mask, token_type_ids, labels = example
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }

    def __len__(self):
        return len(self.all_dataset)


def dataloader_packer(dataset_pd, tokenizer):
    import torch
    from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

    input_ids = []
    attention_masks = []
    token_type_ids = []

    for single_text in dataset_pd.text.values:
        encoded_dict = tokenizer.encode_plus(
            single_text,
            add_special_tokens=True,
            return_token_type_ids=True,
            truncation=True,
            max_length=150,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])
        token_type_ids.append(encoded_dict["token_type_ids"])
        # print(encoded_dict)
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    labels = torch.tensor(dataset_pd.label.values)  # .float()
    tensors = (input_ids, attention_masks, token_type_ids, labels)
    tensors = tensors[:-1] + (tensors[-1].long(),)

    all_dataset = TensorDataset(*tensors)

    dataset = OpinionDataset(all_dataset)

    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=4)
    return dataloader



def log(log_file, text):
    with open(log_file, "a", encoding="utf8") as wfile:
        wfile.write(text + "\n")
