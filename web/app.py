from flask import Flask, request, jsonify, render_template
from web import app


# Get model
# model = ModelInference(tokenizer, rdrsegmenter, r'./model/weights/model.pt')


# API for build UI
@app.route('/', methods=['GET', 'POST'])
def test():
    # if request.method == 'POST':
    #     RATING_ASPECTS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]
    #     # Get review from input
    #     review_sentence = request.form['input_review']
    #     # Predict
    #     predict_results = model.predict(review_sentence)
    #     results = dict()
    #     for i, aspect in enumerate(RATING_ASPECTS):
    #         results.update({aspect: str(predict_results[i]) + " ⭐"})
    #     return render_template("index.html", predict=results)
    #
    # else:
    #     return render_template("index.html")
    return render_template("index.html")


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
    app.run(host='0.0.0.0', port=8000, debug=True)
