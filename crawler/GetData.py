from time import sleep
from selenium.webdriver.common.by import By


def get_URLs(threads, browser, page):
    try:
        browser.get(f"https://agro.gov.vn/vn/p{page}_2Ca-phe.html")
        # browser.get(f"https://agro.gov.vn/vn/p{i}_9Lua-gao.html")
        # browser.get(f"https://agro.gov.vn/vn/p{i}_10Cao-su.html"")
        sleep(threads)
        elements = browser.find_elements(By.CSS_SELECTOR, ".news [href]")
        links = [element.get_attribute('href') for element in elements]
        browser.close()
        print(f"crawl page {page} successfully")
        return [*set(links[0:10])]
    except:
        print(f"get Error at page {page}")


def get_News(threads, browser, url, category=2):
    try:
        browser.get(url)
        sleep(threads)
        title = browser.find_element(By.ID, "ctl00_maincontent_N_TIEUDE")
        date = browser.find_element(By.ID, "ctl00_maincontent_N_NGAYTHANG")
        brief = browser.find_element(By.ID, "ctl00_maincontent_N_TRICHDAN")
        content = browser.find_element(By.ID, "ctl00_maincontent_N_NOIDUNG")
        new_row = [category, title.text, brief.text, date.text, content.text, url]
        browser.close()
        print(f"crawl url: '{url}' successfully")
        return new_row

    except:
        new_row = {'category': category, 'title': "", 'brief': "", 'date': "",
                   'content': "", 'sources': url}
        browser.close()
        print(f"get Error at url: '{url}'")
        return new_row
