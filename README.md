# Predicting the Target Word of Game-playing Conversations using a Low-Rank Dialect Adapter for Decoder Models
**Authors:** Dipankar Srirag and Aditya Joshi and Jacob Eisenstein

**DOI:** [10.48550/arXiv.2409.00358](https://doi.org/10.48550/arXiv.2409.00358)

## Abstract
Dialect adapters that improve the performance of LLMs for NLU tasks on certain sociolects/dialects/national varieties ('dialects' for the sake of brevity) have been reported for encoder models. In this paper, we extend the idea of dialect adapters to decoder models in our architecture called `LoRDD`. Using [`MD-3`]((https://doi.org/10.48550/arXiv.2305.11355)), a publicly available dataset of word game-playing conversations between dialectal speakers, our task is Target Word Prediction (TWP) from a masked conversation. `LoRDD` combines task adapters and dialect adapters where the latter employ contrastive learning on pseudo-parallel conversations from MD-3. Our results for `en-IN` conversations on two models (`Mistral` and  `Gemma`) show that `LoRDD` outperforms four baselines on TWP, while bridging the performance gap with `en-US` by 12% on word similarity and 25% on accuracy. The focused contribution of `LoRDD` is in its promise for dialect adaptation of decoder models.

## Keywords
- Large Language Models
- Dialect Robustness
- Conversation Understanding
- Word-Guessing Game
- Adapters

## BibTeX Citation
<tab><tab>
```bibtex
@inproceedings{srirag-etal-2025-predicting,
    title = "Predicting the Target Word of Game-playing Conversations using a Low-Rank Dialect Adapter for Decoder Models",
    author = "Srirag, Dipankar  and
      Joshi, Aditya  and
      Eisenstein, Jacob",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-short.2/",
    pages = "8--17",
    ISBN = "979-8-89176-190-2"
}
```

## Contact
- [Dipankar Srirag](mailto:d.srirag@unsw.edu.au); [University of New South Wales](https://dipankarsrirag.github.io)
- [Aditya Joshi](mailto:aditya.joshi@unsw.edu.au); [University of New South Wales](https://www.unsw.edu.au/staff/aditya-joshi)
- [Jacob Eisenstein](mailto:jeisenstein@google.com); [Google DeepMind](https://jacobeisenstein.github.io)
