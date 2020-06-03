#NE = non essential

#======================================================================== #
''' importing the transcripts    '''
#======================================================================== #
#function, created from raf's script such that it easily callable



directory_youtubecsv = '/Users/flexonyox/Desktop/CODE/YT_ConSpiracy-master/'

directory_transcripts = '/Users/flexonyox/Desktop/CODE/YT_ConSpiracy-master/Transcripts'

from my_FunctionsModule_ForThesis import TranscriptExtractor #@ tolga, raf; make sure that you put raf's transcript extractor script in a function .py file to import and call it here.

Transcripts = TranscriptExtractor(directory_youtubecsv, directory_transcripts)

transcripts_strings=[]
# labels=[]
# videoId_list = [] #not necessary perse, just to keep track if necessary

for X in Transcripts:
    Transcript, Label,VideoId = X[2], X[3], X[0]
    transcripts_strings.append(Transcript)
    # labels.append(Label)
    # videoId_list.append(VideoId)

# =============================================================================
# Data Exploration
# =============================================================================

#computing the size of the corpus:
from decimal import Decimal # NE: for clarity
print('%.2E' % Decimal (len(str(transcripts_strings))))

#there are some really short transcripts_strings:...
transcripts_lenList = [len(x) for x in transcripts_strings]
transcripts_lenList.sort()
print(len(transcripts_strings[0])) #length of characters of 1st transcript
print(transcripts_lenList, "\n total words summed: ", sum(transcripts_lenList)) #total nr. of characters



# if you want to export to a txt file 
# with open("output_file.txt", "w") as output:
#     output.write(str(transcripts_strings[0]))

#======================================================================== #
'''                 NLP    '''
#======================================================================== #

# =============================================================================
# Creating the spacy Doc file
# =============================================================================

#importing the spacy library and its NLP module for English, which contains, among other things, a tokenizer
import spacy
import en_core_web_sm
#Here I only want to perform tokenizing, so disabling the parser, the tagger, and the named entity recognizer.
nlp = en_core_web_sm.load(disable=["parser", "tagger", "ner"])


# MAKES SPACY WORK WITH LONGER DOCUMENTS:
nlp.max_length=1.1E7
# feeding the transcripts_strings to the pipeline, which obtains a doc object, which contains all the tokens, but also has many other features.
doc = nlp(str(transcripts_strings))

# =============================================================================
# #extracting the tokens from the doc file in a seperate list
# =============================================================================
token_list = [token.text for token in doc]
print(token_list[:100])
print('%.2E' % Decimal ( len(token_list) )) 
print ( len(set(token_list)) ) 
# print (sum(lenList) / len(token_list) ) 

# Can do this too with the lemmas of the tokens:
# lemma_list = [token.lemma_ for token in doc]
# print ( len(lemma_list))
# print ( len(set(lemma_list)) )


# =============================================================================
# Cleaning the tokens
# =============================================================================
def my_cleaner3(text):
	return[token.lemma_.lower() for token in nlp(text) if not (token.is_stop or token.is_alpha==False or len(token.lemma_) <3) ]
cleaned_tokens1 = my_cleaner3(doc.text)

# == :
# #datacleaner method 1: (clearness method)
# cleaned_tokens1 = [token.lemma_.lower() for token in doc if not (token.is_stop or token.is_alpha==False or len(token.lemma_) <3) ] #OR = MORE TOKENS DELETED, (BUT THIS DIDNT SHOW BECAUSE YOU HAVE TO NOT FORGET THE: "()" AFTER IF NOT)
# # print ( len(cleaned_tokens1),len(token_list),'\n deleted tokens:', len(token_list)-len(cleaned_tokens1) )
# len(cleaned_tokens1)
# # len ( set(cleaned_tokens1) )


# =============================================================================
# # analysis of what has been done in the cleaning stage:
# =============================================================================
cleaned_tokens1[:100]
print ('nr of tokens in original uncleaned corpus:', len(token_list) , '\n wat is left after cleaning: ', len(cleaned_tokens1) , '\n tokens that are deleted:',len(token_list)-len(cleaned_tokens1))
# printing some actual tokens that are left after cleaning:
