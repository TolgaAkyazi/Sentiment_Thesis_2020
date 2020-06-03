# -*- coding: utf-8 -*-
"""
Created on Mon May 4 11:12:28 2020
@author: Siebe Albers
"""
#======================================================================== #
'load a df with video ids (which will be used for the youtube api to download the transcripts: and later on for extracting the labels'

with open('IdList_selfWachtedYoutubeVids.txt',encoding="utf-8") as f:
    idList = f.readlines() #txt file with the ids:
#alternatively:
#dic = dict_oldDf # a dictionary where the keys correspond the the youtubeIDs
#======================================================================== #
' downloading the transcripts by their ids                           '
#======================================================================== #
from youtube_transcript_api import YouTubeTranscriptApi
import time # just to record how long it takes to download the transcripts
STARTTIME = time.time() #plus counting the time
Transcripts_w_timestamps = YouTubeTranscriptApi.get_transcripts(video_ids=idList,continue_after_error=True)
Transcripts_w_timestamps = Transcripts_w_timestamps[0]
print('time it took:', time.time() - STARTTIME)
print('len trans', len(Transcripts_w_timestamps))# see how many could be downloaded
# =============================================================================
# # creating a dict with transcripts, Writing to string files to (re)create the transcripts
# =============================================================================
#create a list of video ids, serving as keys for next para
IDLIST = list(Transcripts_w_timestamps.keys())
trans_dic_fromApi = {}
for I in IDLIST:
    TRANS = ""
    trans_dic_fromApi[I] = None
    for J in Transcripts_w_timestamps[I]:
#        print(J['text'])
        TRANS += J['text']
        TRANS += " "
    trans_dic_fromApi[I] = TRANS
#======================================================================== #
' Exporting to disk'
#======================================================================== #
import json
with open('ourWatchedYoutubevidsTranscriptskeys.json', 'w') as fp:
    json.dump(trans_dic_fromApi, fp)
