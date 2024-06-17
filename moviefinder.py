import requests
import pandas as pd
import numpy as np
import re
import sys
import os
import ast
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")

#parameters to updates the movie matrix 
genre_score_low = -7
genre_score_high = 10

def import_movie_df():
    #Loading the Movie_df dataset stored locally, faster way to use the app
    movie_df_file_path = 'data/movie_df.csv'
    movie_df = pd.read_csv(movie_df_file_path)
    return movie_df

def create_indice(movie_df):
    #Construct a map of indices and movie titles
    indices = pd.Series(movie_df.index, index=movie_df['id'])
    return indices

def create_cosine_sim(movie_matrix):
    #generated a cosine sim matrix based on the movie matrix submitter
    cosine_sim = linear_kernel(movie_matrix, movie_matrix)
    return cosine_sim

def genre_matrix(movie_df):
    # Excluding movies that have come up already
    movie_df = movie_df[~movie_df['movie_out']]
    # Extract unique genre IDs from the 'genre_ids' column
    movie_df.loc[:, 'genre_ids_str'] = movie_df['genre_ids'].apply(lambda x: ' '.join(map(str, x)))
    all_genres = set([int(genre_id) for genres in movie_df['genre_ids_str'] for genre_id in re.findall(r'\d+', genres)])
    # Create a DataFrame to store genre matrix
    movie_genre_df = pd.DataFrame()
    # Iterate over each unique genre ID and check if it exists in the 'genre_ids' list for each row
    for genre_id in all_genres:
        movie_genre_df[str(genre_id) + '_genre'] = movie_df['genre_ids_str'].apply(lambda x: 2.0 if str(genre_id) in re.findall(r'\d+', x) else 0.0)
    return movie_genre_df


def create_keyword_matrix(movie_df):
     #creating a matrix that has rows the movies from movie_df and as columns a breakdown of the keywords ids 
    
    # Excluding movies that have come up already
    movie_df = movie_df[~movie_df['movie_out']]
    # Extract all unique keywords
    all_keywords = list(set([int(keyword) for keywords in movie_df['keyword'] for keyword in re.findall(r'\d+', keywords)]))
    # Create a new DataFrame to store the result
    movie_keywords_df = pd.DataFrame({'id': movie_df['id']})
    # Iterate over each unique keyword and check if it exists in the 'keyword' list for each row
    for keyword in all_keywords:
        movie_keywords_df[keyword] = movie_df['keyword'].apply(lambda x: 1 if str(keyword) in re.findall(r'\d+', x) else 0)
    # Concatenate all columns at once
    movie_keywords_df = pd.concat([movie_keywords_df['id'], movie_keywords_df.drop(['id'], axis=1)], axis=1)
    movie_keywords_df = movie_keywords_df.drop(columns= ['id'])
    return movie_keywords_df

def merging_matrix(matrix1, matrix2):
    #function to merge 2 matrix into one
    merged_df = pd.concat([matrix1, matrix2], axis=1)
    merged_df.dropna(inplace=True)
    return merged_df

def overview_matrix(movie_df):
    #create a matrix that has rows the movies from movie_df and as columns a breakdown of the words present in the overview field

    # Excluding movies that have come up already
    movie_df = movie_df[~movie_df['movie_out']]
    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')
    #Replace NaN with an empty string
    movie_df['overview'] = movie_df['overview'].fillna('')
    #Construct the required TF-IDF matrix by fitting and transforming the overview data
    tfidf_matrix = tfidf.fit_transform(movie_df['overview'])
    csr_df = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix)
    return csr_df

def genres_list():
    #new version
    # Define the URL of the API endpoint
    url = "https://api.themoviedb.org/3/genre/movie/list?language=en"
    # Define headers (if needed)
    headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    genres = response.json()
    genres_api_list = genres['genres']
    genre_dict = {}
    for genre in genres_api_list:
        genre_dict[genre['id']] = 0

    return genre_dict




    # #create a list with the all the genre ID
    # # Convert lists to strings
    # movie_df_genre = movie_df['genre_ids'].apply(lambda x: ' '.join(map(str, x)))
    # # Extract unique genre IDs from the 'genre_ids' column
    # all_genres = set([int(genre_id) for genres in movie_df_genre for genre_id in re.findall(r'\d+', genres)])
    # # Initialize a dictionary to store genre IDs with values set to 0
    # genre_mapping = {genre_id: 0 for genre_id in all_genres}
    # return genre_mapping



movie_df = import_movie_df()
genre_scores_list = genres_list()
indices = create_indice(movie_df)
movie_genre_matrix = genre_matrix(movie_df)
movie_overview_matrix = overview_matrix(movie_df)
movie_matrix = merging_matrix(movie_genre_matrix,movie_overview_matrix)
cosine_sim = create_cosine_sim(movie_matrix)
rated_movies_list = []


def adding_keywords_to_df(movie_df):
    #API CALL to retrieve keywords from each movie in the list
    # Define the URL of the API endpoint
    base_url = "https://api.themoviedb.org/3/movie"
    # Define headers (if needed)
    headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
    }
    for index, movie_id in movie_df['id'].items():
        keywords_ids = []
        url = f"{base_url}/{movie_id}/keywords"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            keyword_data = response.json()
            #for each movie store all the keywords ids into a list stored in the column keyboard
            for keyword in keyword_data['keywords']:
                keywords_ids.append(keyword["id"]) 
            if pd.isnull(keyword):
                movie_df.at[index, 'keyword'] = []
            else:
                formatted_keywords = [str(id_) for id_ in keywords_ids]
                movie_df.at[index, 'keyword'] = '[' + ', '.join(formatted_keywords) + ']'
    print('keyword assigned')
    movie_df = movie_df[movie_df['keyword'].apply(lambda x: len(x) > 2)]
    
    return movie_df


def get_movie_dataset():
    # URL of the API endpoint for TMDB
    url = "https://api.themoviedb.org/3/discover/movie"
    # query parameters to get a list of most popular movies on the platform
    params = {
        "include_adult": "False",
        "include_null_first_air_dates": "false",
        "language": "en-US",
        "page": 1,  # Start from page 1
        "sort_by": "popularity.desc"
    }
    #Define headers for authorization
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # List to store data from all pages
    all_data = []
    # Fetch data from all pages, the number of pages is 50, which it means a total of 1000 movies gets exported.
    while params['page'] < 101:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:  #if response is positive
            data = response.json()
            all_data.extend(data['results'])  # Append data from the current page
            total_pages = data['total_pages']
            params['page'] += 1  # Move to the next page
            # Check if we have fetched data from all pages and in case stops the while loop
            if params['page'] > total_pages:
                break
        else:
            print(f"Error: {response.status_code} - {response.text}")
            break
    # Create a DataFrame from the collected data
    movie_df = pd.DataFrame(all_data)
    #deleting movies with no overview, duplicates and movies with missing release data.
    #converting overview into string to then delete it
    print("Df shape before changes is " + str(movie_df.shape))
    movie_df['overview'] = movie_df['overview'].astype(str)
    movie_df = movie_df[movie_df['overview'] != 'nan'] 
    movie_df = movie_df.drop_duplicates(subset=['id'])
    movie_df['release_date'] = movie_df['release_date'].astype(str)
    movie_df = movie_df[movie_df['release_date'].str.len() > 0]

    #removing all columns not in use
    columns_to_remove = ['adult', 'backdrop_path', 'popularity', 'video', 'original_title', 'vote_average', 'vote_count' ]
    movie_df = movie_df.drop(columns=columns_to_remove)
    #adding the keywords id
    movie_df = adding_keywords_to_df(movie_df)
    #create a new column to store whether the movie has been already rated (TRUE) or not (FALSE)
    movie_df['movie_out'] = False
    #save a copy in the folder
    movie_df.to_csv("data/movie_df.csv", index=False)
    print("Data saved to output.csv")
    return movie_df 

def get_top_10_movies():
    #function to retrieve the first 10 movie listed in the movie_df
    columns_selected = ['title', 'id','overview','release_date','poster_path']
    first_10_titles = movie_df[columns_selected].head(10)
    
    return first_10_titles


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(id,rated_movies_list = rated_movies_list, indices = indices,movie_df = movie_df ,cosine_sim = cosine_sim):

    # Get the index of the movie that matches the title
    idx = indices[id]
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:20]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Filter out movies that have been rated
    movie_indices = [i for i in movie_indices if movie_df.at[i, 'id'] not in rated_movies_list]
    

    # Return the top 10 most similar movies

    list_recommendation = movie_df.loc[movie_indices, ['title', 'id','overview','release_date','poster_path']]
    return list_recommendation

def update_genre_list(movie_id, genre_scores_list, score, movie_df = movie_df):
    #getting the list of genre to update for the movie rated
    genres_to_update = movie_df.loc[movie_df['id'] == movie_id, 'genre_ids'].values
    print(genres_to_update)
    genres_ids = ast.literal_eval(genres_to_update[0])
    print(genres_ids)
    # going through the list and updating the score
    for id_genre in genres_ids:
        print(id_genre)
        genre_scores_list[int(id_genre)] += score


def update_movies_matrix_by_genre(genre_scores_list):
    #function that updates the relevant genre column in movie matrix
    global movie_matrix
    for score in genre_scores_list:
        if genre_scores_list[score] < genre_score_low:             
            genre_column_name = str(score)+'_genre'
            movie_matrix[genre_column_name] = 0.0
            genre_scores_list[score] = 0
        elif genre_scores_list[score] > genre_score_high:
            genre_column_name = str(score)+'_genre'
            movie_matrix[genre_column_name] = 2.0
            genre_scores_list[score] = 0
    return genre_scores_list

def answer_no(movie_id, genre_scores_list):
    score_no = -2
    update_genre_list(movie_id, genre_scores_list, score_no)


def answer_yes(movie_id, genre_score_list):
    score_yes = 1
    update_genre_list(movie_id, genre_scores_list, score_yes)

def update_movie_matrix():
    global movie_matrix,cosine_sim
    cosine_sim = create_cosine_sim(movie_matrix)


def get_random_movie_id():
    random_movie_id = np.random.choice(movie_df['id'])
    return random_movie_id

def reset_rated_movies_list():
    global rated_movies_list
    rated_movies_list.clear()
    return rated_movies_list

if __name__ == "__main__":
    print("\n ***Get current MOVIE conditions ***\n")

    print("UNDER THE SEAAAAA")