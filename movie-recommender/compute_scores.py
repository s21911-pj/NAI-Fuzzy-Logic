# Authors:
# Karol Kraus s20687
# Piotr Mastalerz s21911

# environmental instructions
# create venv
#   python3 -m venv venv
# activate venv
#   source venv/bin/activate
# install packages
#   pip3 install -r requirements.txt
# run app
#   python3 main.py


import argparse
import json
import numpy as np


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Compute similarity score')
    parser.add_argument('--user1', dest='user1', required=True,
                        help='First user')
    return parser


# Compute the Euclidean distance score between user1 and user2
def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # Movies rated by both user1 and user2
    common_movies = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    # If there are no common movies between the users, 
    # then the score is 0 
    if len(common_movies) < 2:
        return 0

    squared_diff = []

    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))

    return 1 / (1 + np.sqrt(np.sum(squared_diff)))


# Compute the Pearson correlation score between user1 and user2 
def pearson_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # Movies rated by both user1 and user2
    common_movies = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    num_ratings = len(common_movies)

    # If there are no common movies between user1 and user2, then the score is 0 
    if len(common_movies) < 2:
        return 0

    # Calculate the sum of ratings of all the common movies 
    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])

    # Calculate the sum of squares of ratings of all the common movies 
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in common_movies])

    # Calculate the sum of products of the ratings of the common movies
    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for item in common_movies])

    # Calculate the Pearson correlation score
    Sxy = sum_of_products - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings

    if Sxx * Syy == 0:
        return 0

    return Sxy / np.sqrt(Sxx * Syy)

def get_users(data):
    users = []
    for i in data:
        if i != user1:
            users.append(i)
    return users

def get_scores(data, userList, current_user):
    pearsonScoreList = {}
    eucalideanScoreList = {}

    for user in userList:
        eucalideanScoreList[user] = (euclidean_score(data, current_user, user))
    for user in userList:
        pearsonScoreList[user] = (pearson_score(data, current_user, user))

    eucalideanScoreList = sorted(eucalideanScoreList.items(), key=lambda x: x[1], reverse=True)
    pearsonScoreList = sorted(pearsonScoreList.items(), key=lambda x: x[1], reverse=True)

    return eucalideanScoreList, pearsonScoreList

def get_recommended_movies(data, user, recomended_user):
    recommend_movies = []
    not_recommend_movies = []

    sortedMoviesDesc = (sorted(data[recomended_user[0]].items(), key=lambda x: x[1], reverse=True))
    sortedMoviesAsc = (sorted(data[recomended_user[0]].items(), key=lambda x: x[1], reverse=True))

    for item in sortedMoviesDesc:
        if item[0] not in data[user].keys() and len(recommend_movies) < 5 and item[1] > 7:
            recommend_movies.append(item)

    for item in sortedMoviesAsc:
        if item[0] not in data[user] and len(not_recommend_movies) < 5 and item[1] < 4:
            not_recommend_movies.append(item)

    print("Recommended Movies for", user, "based on:", recomended_user[0])
    for movie in recommend_movies:
        print("Title: ", movie[0], ' with score: ', movie[1])

    print("Not recommended Movies for", user, "based on:", recomended_user[0])
    for movie in not_recommend_movies:
        print("Title: ", movie[0], ' with score: ', movie[1])



if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user1 = args.user1

    ratings_file = 'ratings.json'

    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())

    userList = get_users(data)
    pearson_score_data, euclidean_score_data = get_scores(data,userList, user1)

    print("Pearson correlation Method:")
    get_recommended_movies(data, user1, pearson_score_data[0])
    print("---------------------------")
    print("Euclidean distance Method:")
    get_recommended_movies(data, user1, euclidean_score_data[0])
