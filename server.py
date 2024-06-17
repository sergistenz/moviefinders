from flask import Flask, render_template, request
from moviefinder import get_top_10_movies
from moviefinder import get_recommendations
from moviefinder import rated_movies_list
from moviefinder import genre_scores_list
from moviefinder import answer_no
from moviefinder import answer_yes
from moviefinder import get_random_movie_id
from moviefinder import update_movies_matrix_by_genre
from moviefinder import update_movie_matrix
from moviefinder import reset_rated_movies_list
from waitress import serve


app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/moviefinder')
def get_10_movies():
    list_10_movies = get_top_10_movies()
    movies = list_10_movies.to_dict(orient='records')
    reset_rated_movies_list()
    return render_template(
        "moviefinder.html",
        movies=movies
    )


@app.route('/get_rec_movies/<int:movie_id>')
def get_rec_movies(movie_id):
    global rated_movies_list, list_recommendation
    rated_movies_list.append(movie_id)
    list_recommendation = get_recommendations(movie_id)
    recommended_movies_list = list_recommendation.to_dict('records')
    
    return render_template("recommendations.html", recommended_movies=recommended_movies_list, rated_movies_list=rated_movies_list) #rated_movies=rated_movies

@app.route('/rate_movie', methods=['POST'])
def rate_movie():
    global rated_movies_list, list_recommendation, genre_scores_list

    movie_id = int(request.form.get('movie_id'))
    rating = int(request.form.get('rating'))

    rated_movies_list.append(movie_id)

    print(genre_scores_list)

    if rating == 0:
        #didn't like the movie
        answer_no(movie_id,genre_scores_list)
        print("I don't like the movie")
        #since the movie was not appreciated, the next pick up it will be random
        random_movie_id = get_random_movie_id()
        while random_movie_id in rated_movies_list:
            random_movie_id = get_random_movie_id()
        movie_id = random_movie_id
    elif rating == 1:
        #did like the movie
        answer_yes(movie_id,genre_scores_list)
        print("I like the movie") 
    
    print(rated_movies_list)
    print(genre_scores_list)
    print(len(rated_movies_list))
        
    #here check whether need to refresh the data.
    if len(rated_movies_list) % 5 == 0:
        genre_scores_list = update_movies_matrix_by_genre(genre_scores_list)
        print("LIST HAS BEEN RESET IT")
        print(genre_scores_list)
        update_movie_matrix()

    list_recommendation = get_recommendations(movie_id)
    recommended_movies_list = list_recommendation.to_dict('records')

    return render_template("recommendations.html", recommended_movies=recommended_movies_list, rated_movies_list=rated_movies_list)



if __name__ == "__main__":
    serve(app, host="0.0.0.0", port = 8000)