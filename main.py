import math
import time
from flask import render_template, request, redirect, url_for
from flask_login import login_user, logout_user, current_user

from app import utils, preprocessing, app, login_manager, phobert_model


@app.context_processor
def common_response():
    return {
        'categories': utils.load_categories(),
        'ads': utils.read_json('data/ads.json')
    }


@login_manager.user_loader
def user_load(user_id):
    return utils.get_user_by_id(user_id=user_id)


@app.route('/')
def index():
    return render_template("index.html",
                           news=utils.read_json('data/news.json'))


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


@app.route('/login', methods=['get', 'post'])
def user_signin():
    error_msg = ""
    if request.method.__eq__('POST'):  # check method
        username = request.form.get('username')  # get value from form
        password = request.form.get('password')
        user = utils.check_login(username, password)  # check  (user, pass) match  between form submit and database
        if user:
            login_user(user=user)
            return redirect(url_for('index'))  # direct to index.html
        else:
            error_msg = "Wrong password or username"  # assign variable to show error
    return render_template('user/login.html', error_msg=error_msg)


@app.route('/logout')
def user_logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def user_signup():
    error_msg = ""
    if request.method.__eq__('POST'):
        username = str(request.form.get('username')).strip()
        password = str(request.form.get('password')).strip()
        email = str(request.form.get('email')).strip()
        confirm = str(request.form.get('confirm')).strip()
        try:
            if password.__eq__(confirm) :
                utils.add_user(username=username, password=password, email=email)
                return redirect(url_for('user_signin'))
            else:
                error_msg = "Mật khẩu xác nhận không khớp, vui lòng kiểm tra lại"
        except Exception as ex:
            error_msg = "Error: " + str(ex)
    return render_template('user/register.html', error_msg=error_msg)


@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    if not current_user.is_authenticated:
        return render_template('user/login.html')

    if request.method.__eq__('POST'):
        headline = request.form['headline']
        brief = request.form['brief']
        news_sentence = f"{headline} {brief}"  # Get review from input
        start_time = time.time()
        news_sentence = preprocessing.pipeline(news_sentence)
        predict_results = phobert_model.predict(news_sentence)
        final_output = preprocessing.output(predict_results[0])
        print(f"predict successfully - {round(time.time() - start_time, 2)} second")
        return render_template("analysis.html",
                               predict=final_output['results'],
                               sententence={"headline": headline,
                                            "brief": brief})
    else:
        return render_template("analysis.html")


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8080, debug=False)
    app.run()
    # print(phobert_model.predict("giá lúa tăng giá giá lúa tăng giá"))
