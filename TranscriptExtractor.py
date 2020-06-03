import os, json, csv

def TranscriptExtractor(directory_youtubecsv, directory_transcripts):
    "This Function extracts a list from the dataset with the following string contents: [Video_ID, Video_Category, Video_Transcript, Video_Rating]"

    # first we load in all info from the youtube.csv we need
    os.chdir(directory_youtubecsv)
    with open('YouTube.csv', mode='r') as infile:
        reader = csv.reader(infile)
    youtube_ranked = []
    for rows in reader:
        youtube_ranked.append(rows)
    youtube_ranked_data = {}

    for listed in youtube_ranked:
        youtube_ranked_data.update({listed[1][32:]: listed[5]})

    ## now for the transcript data
    id_cat_transcripts = []

    os.chdir(directory_transcripts)

    # firearms
    with open('firearms_transcript.json') as f:
        data_transcript_firearms = json.load(f)
    # here we fill up a dataset with: [ID, Category, Transcript, Final rating] for firearms
    id_transcripts_firearms = list(data_transcript_firearms.keys())
    for id in id_transcripts_firearms:
        id_cat_transcripts.append([id, "firearms", data_transcript_firearms[id], youtube_ranked_data[id]])

    # fitness
    with open('fitness_transcript.json') as f:
        data_transcript_fitness = json.load(f)
    # here we fill up a dataset with: [ID, Category, Transcript, Final rating] for fitness
    id_transcripts_fitness = list(data_transcript_fitness.keys())
    for id in id_transcripts_fitness:
        id_cat_transcripts.append([id, "fitness", data_transcript_fitness[id], youtube_ranked_data[id]])

    # gurus
    with open('gurus_transcript.json') as f:
        data_transcript_gurus = json.load(f)
    # here we fill up a dataset with: [ID, Category, Transcript, Final rating] for fitness
    id_transcripts_gurus = list(data_transcript_gurus.keys())
    for id in id_transcripts_gurus:
        id_cat_transcripts.append([id, "gurus", data_transcript_gurus[id], youtube_ranked_data[id]])

    # martial_arts
    with open('martial_arts_transcript.json') as f:
        data_transcript_martial_arts = json.load(f)
    # here we fill up a dataset with: [ID, Category, Transcript, Final rating] for fitness
    id_transcripts_martial_arts = list(data_transcript_martial_arts.keys())
    for id in id_transcripts_martial_arts:
        id_cat_transcripts.append([id, "martial_arts", data_transcript_martial_arts[id], youtube_ranked_data[id]])

    # natural foods
    with open('natural_foods_transcript.json') as f:
        data_transcript_natural_foods = json.load(f)
    # here we fill up a dataset with: [ID, Category, Transcript, Final rating] for fitness
    id_transcripts_natural_foods = list(data_transcript_natural_foods.keys())
    for id in id_transcripts_natural_foods:
        id_cat_transcripts.append([id, "natural_foods", data_transcript_natural_foods[id], youtube_ranked_data[id]])

    # tiny houses
    with open('tiny_houses_transcript.json') as f:
        data_transcript_tiny_houses = json.load(f)
    # here we fill up a dataset with: [ID, Category, Transcript, Final rating] for fitness
    id_transcripts_tiny_houses = list(data_transcript_tiny_houses.keys())
    for id in id_transcripts_tiny_houses:
        id_cat_transcripts.append([id, "tiny_houses", data_transcript_tiny_houses[id], youtube_ranked_data[id]])

    ## we now have a full set of 600 entries containing all video ID's, categories, transcripts and rankings
    # print(id_cat_transcripts)

    # NOW! let's remove all empty transcripts
    id_cat_transcripts_emptycleaned = []

    for unit in id_cat_transcripts:
        if unit[2] == '':
            continue
        else:
            id_cat_transcripts_emptycleaned.append(unit)

    return id_cat_transcripts_emptycleaned

