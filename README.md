# lat_poetry_gen

Renaissance writers have been fascinated by ancient traditions in writing for a long time and pursued a path to revive an ancient world of creative writing. This project followed a similar goal, not by taking up a pen and paper, but by teaching Latin to a computer. 

It presents VIRGO-1, an early-stage poetry generation system (PGS) designed to produce Latin poems from prosaic input. In total, eleven models were trained: five trained on a 10 million-word corpus, five others on the 22.5 million-word version, and lastly, one trained on 290,000 words of poetry data for comparative analysis. 

While the system shows initial potential, it faces several challenges, particularly in generating coherent and metrically accurate poetry. Despite these hurdles, the system lays a promising foundation for future research, offering insights into a constraint-based approach using prose input and a self-created dictionary for various sizes of vocabularies and two corpus sizes. This work also produced three cleaned datasets and an unduplicated 140 million-word version of the Corpus Corporum, which can support further advancements in Latin poetry generation. I also carried out an online expert validation with 26 Latin experts from Belgium and Germany, which can provide useful pointers for future developments.

The repository contains the following folders and files:

(1)	Word_language model: This makes a reference to the PyTorch infrastructure given by: https://github.com/pytorch/examples/tree/main/word_language_model. Which has almost the same folder structure.
However, many files were added and changed as you will see. Another data set was used (Latin instead of English) for example. But also, additional constraints were added to alter the output generations according to five sub-experiments:
(a)	For the prose data (With CLTK/without/elision recognition/five dactyl requirement) and (b) For poetry (with CLTK/without)

(2)	Keras_model: 
•	General statistics: some stats on the prose and poetry input texts
•	Latin_LM_Pipeline: Mainly, the first half is relevant because that’s about the cleaning of the prose data
•	Latin_LM_pipline_10mio_eos: This is the py file used to train a keras GRU on the next word prediction task for Latin using the 10 million-word corpus and eos tokens.
•	poesia_latina: This is the Excel sheet with all the web-scraped texts from http://www.poesialatina.it/
•	scraping_latina_poesia: web scraping as mentioned above
•	ten_million: contains the cleaned 10 million-word corpus.
o	vocab_10k: This can be ignored since the output was not compared in the thesis. GRU trained with keras with a vocabulary of 10k. The best version has to be looked up in the results table. I uploaded the last six epochs each before the early stopping.
o	vocab_50k: This can be ignored since the output was not compared in the thesis. GRU trained with keras with a vocabulary of 50k.
•	twenty_million: contains the cleaned 22.5 million-word corpus.
