from flask import Flask, request, render_template
from web import *
import math
import utils




@app.context_processor
def common_response():
    return {
        'categories': utils.load_categories()
    }


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/news/<int:product_id>/')
def news_list(product_id):
    return render_template('news.html')


@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    if request.method == 'POST':
        NEWS_ASPECTS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]
        # Get review from input
        review_sentence = request.form['input_review']
        # Predict
        predict_results = phobert_model.predict(review_sentence)
        results = dict()
        for i, aspect in enumerate(NEWS_ASPECTS):
            results.update({aspect: str(predict_results[i]) + " ⭐"})
        return render_template("analysis.html", predict=results)
    return render_template("analysis.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    # app.run(debug=True)
