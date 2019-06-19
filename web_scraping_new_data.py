"""
911 Emergency Calls Forecasting Model
Web Scraping Script
@author: Monika Scislo
"""
from lxml import html
import requests
import time

f= open("14-06-data.txt","w+")
for i in range (1,360):
    from bs4 import BeautifulSoup
    page = requests.get('https://webapp02.montcopa.org/eoc/cadinfo/wap.asp')
    soup = BeautifulSoup(page.text, "html.parser")
    tree = html.fromstring(page.content)
    em = tree.xpath('//br/following::text()')
    one_a_tag = soup.findAll('p')
    par = []
    for x in one_a_tag:
        par.append(str(x))
    with open('14-06-data', "a") as text_file:
        for item in par:
            text_file.write("%s\n" % item)
    time.sleep(240)
    
