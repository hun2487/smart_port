from turtle import color
from flask import Flask, request, json, jsonify
from flask import Flask, render_template
from flask import Flask, redirect, url_for

import pymysql

import pandas as pd
from sqlalchemy import create_engine
from PIL import Image
import base64
from io import BytesIO

import cv2

smart_port_db = pymysql.connect(
    user='admin',
    passwd='admin', 
    host='3.35.222.169', 
    db='smart_port', 
    charset='utf8'
)

cursor = smart_port_db.cursor()

sql = "SELECT * FROM w_log"

cursor.execute(sql)

data_list = cursor.fetchall()

app = Flask(__name__)
@app.route("/", methods=['GET','POST'])
def index():
    return render_template('index.html', data_list=data_list)
@app.route("/image/<index>" )
def image(index):

    engine = create_engine('mysql+pymysql://admin:admin@3.35.222.169:3306/smart_port', echo=True)

    sql = f'select * from w_log where image_index = {index} '
    img_df = pd.read_sql(sql=sql, con=engine)

    ship_img_str = img_df['ship_image'].values[0]
    ship_img = base64.decodestring(ship_img_str)

    im = Image.open(BytesIO(ship_img))
    with BytesIO() as buf:
        im.save(buf, 'jpeg')
        image_bytes = buf.getvalue()
    ship_img = base64.b64encode(image_bytes).decode()

    person_img_str = img_df['person_image'].values[0]
    person_img = base64.decodestring(person_img_str)

    im = Image.open(BytesIO(person_img))
    with BytesIO() as buf:
        im.save(buf, 'jpeg')
        image_bytes = buf.getvalue()
    person_img = base64.b64encode(image_bytes).decode()
    return render_template('index.html', data_list=data_list, ship_image = ship_img, person_image = person_img), 200

ship_count = 0
p_count = 0
p_degree = 0
@app.route('/ship', methods=['POST'])
def test():
    global ship_count
    params1 = request.get_json()
    ship_count = params1.get('count')
    n_count = params1.get('n_count')
    print('선박 수', ship_count)
    #print('n_count', n_count)
    if ship_count>0 and p_count == 0 :
        response = {
            'result': False
        }
    elif ship_count>0 and p_degree<=-10 :
        response = {
            'result' : False
        }
    elif p_count ==0 :
        response = {
            'result' : False
        }
    else :
        response = {
            'result' : True
        }
    print(response)
    return jsonify(response)

@app.route('/person', methods=['POST'])
def person():
    global p_count, p_degree
    params2 = request.get_json()
    p_count = params2.get('p_count')
    p_degree = params2.get('degree')
    print('사람 수',p_count,'얼굴 각도', p_degree)
    if ship_count>0 and p_count == 0 :
            response = {
                'result': False
            }
    elif ship_count>0 and p_degree<=-10 :
            response = {
                'result' : False
            }
    elif p_count ==0 :
            response = {
                'result' : False
            }
    else :
            response = {
                'result' : True
            }

    return jsonify(response)


if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
