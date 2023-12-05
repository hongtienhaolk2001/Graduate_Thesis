import math

from flask import request, render_template

from app import utils, preprocessing, app, phobert_model


@app.context_processor
def common_response():
    return {
        'categories': utils.load_categories()
    }


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/news/<int:category_id>/', methods=['GET', 'POST'])
def news_list(category_id=0):
    category_name = {0: " ", 1: "Lúa Gạo", 2: "Cà Phê", 3: "Cao Su"}
    page = request.args.get('page', 1)
    kw = request.args.get('keyword')
    news = utils.load_news(category_id=category_id, page=int(page), keyword=kw)
    counter = utils.count_news(category_id)
    return render_template('news.html',
                           category_name=category_name[category_id],
                           cate_id=category_id,
                           news=news,
                           pages=math.ceil(counter / app.config['PAGE_SIZE']))


@app.route('/read/')
def read():
    return render_template('read.html')


@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    if request.method.__eq__('POST'):
        news_sentence = f"{request.form['headline']} {request.form['brief']}"  # Get review from input
        news_sentence = preprocessing.pipeline(news_sentence)
        predict_results = phobert_model.predict(news_sentence)
        final_output = preprocessing.output(predict_results[0])
        return render_template("analysis.html",
                               predict=final_output['results'])
    else:
        return render_template("analysis.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
