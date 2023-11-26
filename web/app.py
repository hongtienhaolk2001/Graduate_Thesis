from flask import Flask, request, render_template
from web import app, phobert_model
import utils
import NewsPreprocessing
import math


@app.context_processor
def common_response():
    return {
        'categories': utils.load_categories()
    }


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/news/<int:category_id>/')
def news_list(category_id):
    category_name = {1: "Lúa Gạo", 2: "Cà Phê", 3: "Cao Su"}
    page = request.args.get('page', 1)
    news = utils.load_news(category_id=category_id, page=int(page))
    counter = utils.count_news(category_id)
    return render_template('news.html',
                           category_name=category_name[category_id],
                           cate_id=category_id,
                           news=news,
                           pages=math.ceil(counter / app.config['PAGE_SIZE']))


@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    if request.method == 'POST':
        NEWS_ASPECTS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]
        news_sentence = NewsPreprocessing.pipeline(request.form['input_news'])  # Get review from input
        predict_results = phobert_model.predict(news_sentence)  # Predict
        results = dict()
        for i, aspect in enumerate(NEWS_ASPECTS):
            results.update({aspect: str(predict_results[i]) + " ⭐"})
        return render_template("analysis.html", predict=results)
    return render_template("analysis.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    # app.run(debug=True)
