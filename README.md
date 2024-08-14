**Important:** Due to the size of the files/ a server error, the remaining files are available on the drive: https://drive.google.com/drive/folders/1p_zk4upxpaCgEbMqDRK6pSBPAxnuXdog?usp=sharing. 

# lat_poetry_gen
Renaissance writers have long been fascinated by ancient traditions in writing and pursued a path to revive an ancient world of creative writing. This project follows a similar goal, not by taking up a pen and paper, but by teaching Latin to a computer.

It presents VIRGO-1, an early-stage poetry generation system (PGS) designed to produce Latin poems from prosaic input. In total, eleven models were trained:
- Five models were trained on a 10 million-word corpus.
- Five others were trained on a 22.5 million-word corpus.
- One model was trained on 290,000 words of poetry data for comparative analysis.
  
While the system shows initial potential, it faces several challenges, particularly in generating coherent and metrically accurate poetry. Despite these hurdles, the system lays a promising foundation for future research, offering insights into a constraint-based approach using prose input and a self-created dictionary for various vocabulary sizes and two corpus sizes. This work also produced three cleaned datasets:
- An unduplicated 140 million-word version of the Corpus Corporum, which can support further advancements in Latin poetry generation with version 10 million words and 20 million words
- A poetry collection with different meters or only the dactylic hexameter.

The repository contains the following folders and files:
- Word_language_model: This references the PyTorch infrastructure available at: https://github.com/pytorch/examples/tree/main/word_language_model, which has a similar folder structure.
Many files were added and changed. For instance, another dataset (Latin instead of English) was used, and additional constraints were added to alter output generations according to five sub-experiments:
  - (a) For prose data: With CLTK/without/elision recognition/five dactyl requirement.
  -  (b) For poetry: With CLTK/without.
- Keras_model:
  - General statistics: Contains some statistics on the prose and poetry input texts.
  - Latin_LM_Pipeline: The first half is mainly relevant as it covers the cleaning of the prose data.
  - Latin_LM_pipeline_10mio_eos: The Python file used to train a Keras GRU on the next word prediction task for Latin using the 10 million-word corpus and EOS tokens.
  - poesia_latina: An Excel sheet with all the web-scraped texts from http://www.poesialatina.it/.
  - scraping_latina_poesia: The web scraping script as mentioned above.
  - ten_million: Contains the cleaned 10 million-word corpus.
  -   vocab_10k: Can be ignored since the output was not compared in the thesis. GRU trained with Keras with a vocabulary of 10k. The best version can be found in the results table. The last six epochs before early stopping are uploaded.
  -   vocab_50k: Can be ignored since the output was not compared in the thesis. GRU trained with Keras with a vocabulary of 50k.
  - twenty_million: Contains the cleaned 22.5 million-word corpus.
