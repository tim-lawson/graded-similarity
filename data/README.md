# SemEval-2020 Task 3: Graded Word Similarity in Context

## Task Description:

For this tasks we asked participants to build systems that try to predict the effect that context has in human perception of similarity of words.

We have seen very interesting work that uses local context to predict discrete changes in meaning: the different senses of a polysemous word. However context also has more subtle, continuous (graded) effects on meaning, even for words not necessarily considered polysemous.

In order to be able to look at these effects we are building several datasets where we ask annotators to score how similar a pair of words are after they have read a short paragraph (which contains the two words). 
Each pair is scored within two of these paragraphs, allowing us to look at changes in similarity ratings due to context.

For more details:
	- LREC2020 paper -> https://www.aclweb.org/anthology/2020.lrec-1.720
	- Semeval-2020 Task3 -> https://competitions.codalab.org/competitions/20905
	- A task description paper will be published in the Proceedings of the 14th International Workshop on Semantic Evaluation. 

## Files Included:

1. cosimlex_dataset.zip
2. evaluation_kit_final.zip
3. practice_kit_final.zip

The first file included is the CoSimLex dataset, which was used as the gold standard results for this task.
The second and the third are the evaluation and practice kits that were provided to participants.
All of them contain more detailed text files explaining their respective contents.

## References

Please make sure you reference our work if you make use of this data:

@inproceedings{armendariz-etal-2020-semeval,
    title = "{SemEval-2020} Task 3: Graded Word Similarity in Context ({GWSC})",
    author = "Armendariz, Carlos S.  and
      Purver, Matthew  and
      Pollak, Senja  and
      Ljube{\v{s}}i{\'{c}}, Nikola  and
      Ul{\v{c}}ar, Matej  and
      Robnik-{\v{S}}ikonja, Marko and
      Vuli{\'{c}}, Ivan and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 14th International Workshop on Semantic Evaluation",
    year = "2020",
    address="Online"
}

@InProceedings{armendariz-EtAl:2020:LREC,
	author    = {Armendariz, Carlos S.  and  Purver, Matthew  and  Ulčar, Matej  and  Pollak, Senja  and  Ljubešić, Nikola  and  Granroth-Wilding, Mark},
  	title     = {{CoSimLex}: A Resource for Evaluating Graded Word Similarity in Context},
	booktitle      = {Proceedings of The 12th Language Resources and Evaluation Conference},
	month          = {May},
	year           = {2020},
	address        = {Marseille, France},
	publisher      = {European Language Resources Association},
	pages     = {5878--5886},
	url       = {https://www.aclweb.org/anthology/2020.lrec-1.720}
} 

## Contact: 
Carlos Armendariz
c.santosarmendariz@qmul.ac.uk
