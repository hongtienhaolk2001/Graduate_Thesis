from flask import Flask, request, jsonify, render_template
from web import *
import math
import utils


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/analysis')
def analysis():
    return render_template("analysis.html")


@app.route('/news/<int:product_id>/')
def news_list(product_id):
    page = request.args.get('page', 1)
    counter = utils.count_product()
    return render_template('news.html',
                           product=int(product_id),
                           news_list=utils.load_news(product_id=product_id, page=int(page)),
                           pages=math.ceil(counter / app.config['PAGE_SIZE'])
                           )


# API for build UI
# @app.route('/analysis', methods=['GET', 'POST'])
# def analysis():
#     if request.method == 'POST':
#         RATING_ASPECTS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]
#         # Get review from input
#         review_sentence = request.form['input_review']
#         # Predict
#         predict_results = model.predict(review_sentence)
#         results = dict()
#         for i, aspect in enumerate(RATING_ASPECTS):
#             results.update({aspect: str(predict_results[i]) + " ⭐"})
#         return render_template("index.html", predict=results)
#
#     else:
#         return render_template("analysis.html")

# API for submit
# @app.route("/review-solver/solve")
# def solve():
#     RATING_ASPECTS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]
#
#     # Get reviews
#     review_sentence = request.args.get('review_sentence')
#     # Model predict
#     predict_results = model.predict(review_sentence)
#     output = {
#         "review": review_sentence,
#         "results": {}
#     }
#     # Return json
#     for count, r in enumerate(RATING_ASPECTS):
#         output["results"][r] = int(predict_results[count])
#     return jsonify(output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    # app.run(debug=True)
