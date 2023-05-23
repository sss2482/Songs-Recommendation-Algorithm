import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity




def loaddata(songs_url,user_profile_url,likes_url):
    global metadata,songs_data,user_data,likes_data

    songs_url='https://drive.google.com/uc?id=' + songs_url.split('/')[-2]
    user_profile_url='https://drive.google.com/uc?id=' + user_profile_url.split('/')[-2]
    likes_url='https://drive.google.com/uc?id=' + likes_url.split('/')[-2]
    metadata = pd.read_csv(songs_url)
    songs_data=pd.read_csv(songs_url)
    user_data= pd.read_csv(user_profile_url)
    likes_data=pd.read_csv(likes_url)

def load_data(songs_url,user_profile_url,likes_url):
    global metadata,songs_data,user_data,likes_data
    metadata = pd.read_csv(songs_url)
    songs_data=pd.read_csv(songs_url)
    user_data= pd.read_csv(user_profile_url)
    likes_data=pd.read_csv(likes_url)


#convert genres in list form
l_genres=[]
def to_list(x):
    global l_genres
    l=list(str(x).split(','))

    for i in l:
        if i not in l_genres:
            l_genres.append(i)
    return l
def to_list_2(x):
    l=list(str(x).split(','))
    return l

# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


#Function that creates string mixture of preferences
def create_soup(x):
    return  ' '.join(x['genres'])

def find_genre(x,i):

    if i in x['genres']:
        return 1
    if i in x['language']:
        return 1
    if i in x['artist']:
        return 1

#Function that adds score of first algorithm to dataframe
def set_score_1():
    # Get the similarity scores 
    sim_scores = list(enumerate(cosine_sim[len(cosine_sim)-1]))
    sim_scores=sim_scores[:-1]
    # Sort the songs based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the song indices
    indices = [i[0] for i in sim_scores]

    # Add score_1 to metadata
    metadata['score_1']=0
    for i in sim_scores:
        metadata.loc[sim_indices[i[0]],'score_1']=i[1]

#Function that adds score of first algorithm to dataframe
def set_score_2():
    for i in range(len(user_predicted_like)):
        metadata.loc[i,'score_2']=user_predicted_like[i]

def run_algorithm_1(userId):

    global sim_indices, song_indices, cosine_sim
    features=['genres',]
    for feature in features:
        metadata[feature]=metadata[feature].apply(to_list)
        metadata[feature] = metadata[feature].apply(clean_data)
    metadata['soup'] = metadata.apply(create_soup, axis=1)
    #user-profile of preferences
    user_item= user_data.loc[userId]
    user_languages=to_list_2(user_item['languages'])
    user_genres=to_list(user_item['genres'])
    user_genres=clean_data(user_genres)

    #string mixture of user preferences
    user_soup=' '.join(user_genres)
    
    #metadata for first algorithm(considered only user preferred languages)
    metadata_1=metadata[metadata['language'].isin(user_languages) ]
    metadata_1=metadata_1.reset_index()
    metadata_1.loc[len(metadata_1),['soup']]=user_soup

    #counting the occurrences of words in the soup
    count = CountVectorizer()
    count_matrix = count.fit_transform(metadata_1['soup'])

    #matrix showing similarity among songs and also between the user based on cosine similarity
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    #indices to get songIds
    sim_indices = pd.Series(metadata_1['songId'],index=metadata_1.index)
    song_indices=pd.Series(metadata['title'], index=metadata.index)

    set_score_1()

def run_algorithm_2(userId):
    global like, user_predicted_like

    #Algorithm 2 based on activity of user
    features = ['language','artist']
    for feature in features:
        metadata[feature]=metadata[feature].apply(to_list)
        metadata[feature] = metadata[feature].apply(clean_data)
    #Add each genre to dataframe
    for i in l_genres:
        i=i.lower()
        metadata[i]=metadata.apply(find_genre, args=(i,), axis=1)
    
    metadata.sort_values('songId', inplace=True)
    metadata.set_index('songId', inplace=True)

    #Merge likes and metadata
    metadata_merged = pd.merge(likes_data, metadata, on='songId', how='outer')

    #Create a matrix of likes between userId and songId
    like = pd.pivot_table(metadata_merged, values='like', index=['songId'], columns = ['userId'], dropna = False, fill_value=0)
    like.sort_index(axis=1, inplace=True)

    users_no = like.columns

    metadata_genre=metadata.copy()
    metadata_genre.drop(['genres','soup','title','artist','language','score_1'], axis=1, inplace=True)
    # print(metadata_genre)

    working_metadata = metadata_genre.mul(like.iloc[:,userId], axis=0)
    working_metadata.replace(0, np.NaN, inplace=True)    
    metadata_users = working_metadata.sum(axis=0)


    document_frequency = metadata_genre.sum()

    imetadata = 1/(document_frequency+1)

    #The dot product of article vectors and Imetadata vectors gives us the weighted scores of each article.
    imetadata_metadata = metadata_genre.mul(imetadata)
    working_metadata = imetadata_metadata.mul(metadata_users, axis=1)
    user_predicted_like = working_metadata.sum(axis=1)

    #Normalising the predicted like to have the value less between 0 to 1
    user_predicted_like=user_predicted_like/user_predicted_like.sum()

    set_score_2()

def songs_liked_by_user(userId):
    user_df=like[like.index== userId]
    songs_liked=user_df.columns[user_df.values[0]==1].to_list()

    return songs_liked

#Function that runs the third algorithm
def run_algorithm_3(user):
    global like
    like=like.T

    songs_liked=songs_liked_by_user(user)

    songs_liked_df = like[songs_liked]

    user_song_count = songs_liked_df.T.sum()

    user_song_count = user_song_count.reset_index()
    user_song_count.columns = ["userId","song_count"]
    user_song_count['jaq_ind']=user_song_count['song_count']/len((songs_liked_by_user(user_song_count['userId'])))

    perc=(30/100)
    user_song_count= user_song_count[user_song_count["jaq_ind"] > perc]

    top_users_likes =user_song_count.merge(likes_data[["userId", "songId", "like"]], how='inner')

    top_users_likes = top_users_likes[top_users_likes["userId"] != user]

    top_users_likes['weighted_like'] = top_users_likes['jaq_ind'] * top_users_likes['like']


    recommendation_df = top_users_likes.groupby('songId').agg({"weighted_like": "mean"})
    recommendation_df = recommendation_df.reset_index()
    weighted_song_like=pd.Series(recommendation_df['weighted_like'],index=recommendation_df['songId'])
    metadata['score_3']=0
    for i in recommendation_df['songId']:
        metadata.loc[i,'score_3']=weighted_song_like[i]


#Funtion that returns overall recommended songs
def recommend(userId,n,songs_url,user_profile_url,likes_url):
    #loaddata(songs_url,user_profile_url,likes_url)
    load_data(songs_url,user_profile_url,likes_url)

    #run algorithms
    run_algorithm_1(userId)
    run_algorithm_2(userId)
    run_algorithm_3(userId)
    #Add scores of both algorithms
    metadata['score']=metadata['score_1']+metadata['score_2']+(metadata['score_3']/2)

    # Make the scores of already listened songs to zero
    user_like=like.T[userId]
    for i in range(len(user_like)):
        if user_like[i]==1:
            metadata.loc[i,'score']=0
    
    #Sort the songs according to score
    metadata.sort_values(by='score', inplace=True, ascending=False)

    #return the recommended songs
    return list(metadata['title'].iloc[0:n])
print(recommend(0,15,'data/songs_main.csv','data/user_profile.csv','data/likes.csv'))


def recommended_songs_song(song_Id, n):
  song_name = like[song_Id]
  songs_from_item_based = like.corrwith(song_name).sort_values(ascending=False)
  return songs_from_item_based[1:n].index.to_list()

# print(recommended_songs_song(3,5))



    


