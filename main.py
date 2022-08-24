from flask import Flask, redirect, url_for, render_template, request, jsonify 
import pandas as pd
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn import tree
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeClassifier
import graphviz
from math import log
import random
import json

import os
app = Flask(__name__)

BASE_URL = "http://192.168.55.12:5000/"

@app.route('/')
def index():
    dr = {'BASE_URL' : BASE_URL}
    stLogin = {'stLogin' : 'no'}
    return render_template('home.html', dRes=dr, stLogin=stLogin)

@app.route('/login')
def login():
    dr = {'BASE_URL' : BASE_URL}
    return render_template('login.html', dRes=dr)

@app.route('/proses-login', methods=('GET', 'POST'))
def proses_login():
    # {'username':username, 'password':password}
    username = request.form['username']
    password = request.form['password']

    dataUser = getUserJson()

    if username == dataUser['username'] and password == dataUser['password']:
        sLogin = 'sukses'
    else:
        sLogin = 'gagal'

    dr = {'status' : sLogin}
    return jsonify(dr)

@app.route('/data-siswa')
def data_siswa():
    dr = {'BASE_URL' : BASE_URL}
    sLogin = cekLogin()
    if sLogin == "no":
        stLogin = {'stLogin' : 'no'}
        return render_template('home.html', dRes=dr, stLogin=stLogin)
    else:
        dSiswa = []
        dataSiswa = pd.read_excel("./DATA_TRAINING.xlsx")
        dtnp = dataSiswa.to_numpy()
        ord = 1

        for x in dtnp:
            dSatuan = {}
            dSatuan['nama'] = x[1]
            dSatuan['ord'] = ord
            dSatuan['kelas'] = x[2]
            dSatuan['penyampaian_materi'] = x[3]
            dSatuan['media_pembelajaran'] = x[4]
            dSatuan['suasana_belajar'] = x[5]
            dSatuan['tugas'] = x[6]
            dSatuan['kehadiran'] = x[7]
            dSatuan['praktikum'] = x[8]
            dSatuan['uts' ] = x[9]
            dSatuan['uas'] = x[10]
            dSatuan['matematika'] = x[11]
            dSatuan['b_indo'] = x[12]
            dSatuan['b_inggris'] = x[13]
            dSatuan['pemahaman'] = x[14]
            dSiswa.append(dSatuan)
            ord += 1

        dr = {'BASE_URL' : BASE_URL}
        return render_template('data-siswa.html', dSiswa=dSiswa)

@app.route('/normalisasi-data-training')
def normalisasi_data_training():
    dSiswa = []
    dataSiswa = pd.read_excel("./DATA_TRAINING.xlsx")
    dataSiswa.replace({'penyampaian_materi':{'Serius Santai':4, 'Serius':3, 'Santai':2, 'Membosankan':1}},inplace=True)
    dataSiswa.replace({'media_pembelajaran':{'Ebook':4, 'Video':3, 'PPT':2, 'PDF':1}},inplace=True)
    dataSiswa.replace({'suasana_belajar':{'Mendukung':4, 'Tidak Mendukung':1}},inplace=True)
    dataSiswa.replace({'tugas':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswa.replace({'kehadiran':{'Sangat Baik':4, 'Baik':3, 'Cukup':2}},inplace=True)
    dataSiswa.replace({'praktikum':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswa.replace({'uts':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswa.replace({'uas':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswa.replace({'matematika':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswa.replace({'bindo':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswa.replace({'bing':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswa.replace({'pemahaman':{'Tinggi':1, 'Rendah':0}},inplace=True)
    dtnp = dataSiswa.to_numpy()
    ord = 0

    for x in dtnp:
        dSatuan = {}
        dSatuan['nama'] = x[1]
        dSatuan['ord'] = ord
        dSatuan['kelas'] = x[2]
        dSatuan['penyampaian_materi'] = x[3]
        dSatuan['media_pembelajaran'] = x[4]
        dSatuan['suasana_belajar'] = x[5]
        dSatuan['tugas'] = x[6]
        dSatuan['kehadiran'] = x[7]
        dSatuan['praktikum'] = x[8]
        dSatuan['uts' ] = x[9]
        dSatuan['uas'] = x[10]
        dSatuan['matematika'] = x[11]
        dSatuan['b_indo'] = x[12]
        dSatuan['b_inggris'] = x[13]
        dSatuan['pemahaman'] = x[14]
        dSiswa.append(dSatuan)
        ord += 1

    # proses data training 
    totalRecord = ord
    # data pemahaman 
    dp = {'tinggi':0, 'rendah':0, 'total':0, 'entropy':0}
    dpT = dataSiswa['pemahaman'] == 1
    dpR = dataSiswa['pemahaman'] == 0
    dp['tinggi'] = dpT.sum()
    dp['rendah'] = dpR.sum()
    dp['total'] = totalRecord
    dp['entropy'] = cse(dp['tinggi'], dp['total']) + cse(dp['rendah'], dp['total'])

    # penyampaian materi
    pm = {
        'serius' : {'tinggi': 0, 'rendah' : 0, 'entropy':0},
        'santai' : {'tinggi': 0, 'rendah' : 0, 'entropy':0},
        'serius_santai' : {'tinggi': 0, 'rendah' : 0, 'entropy':0},
        'membosankan' : {'tinggi': 0, 'rendah' : 0, 'entropy':0},
        'gain' : 0
    }
    # media pembelajaran 
    mp = {
        'pdf' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'video' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'ppt' :  {'tinggi':0, 'rendah':0, 'entropy':0},
        'ebook' :  {'tinggi':0, 'rendah':0, 'entropy':0},
        'gain' : 0
    }
    # suasana belajar 
    sb = {
        'mendukung' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'tidak_mendukung' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'gain' : 0
    }
    # tugas 
    tg = {
        'baik' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'cukup' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'sangat_baik' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'kurang' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'gain': 0
    }
    # kehadiran 
    kh = {
        'baik' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'sangat_baik' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'cukup' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'gain' : 0
    }
    # praktikum 
    pk = {
        'baik' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'sangat_baik' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'cukup' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'kurang' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'gain' : 0
    }
     # uts 
    uts = {
        'baik' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'sangat_baik' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'cukup' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'kurang' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'gain': 0
    }
    # uas 
    uas = {
        'baik' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'sangat_baik' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'cukup' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'kurang' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'gain': 0
    }
     # matematika 
    mat = {
        'baik' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'sangat_baik' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'cukup' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'kurang' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'gain': 0
    }
     # bahasa indonesia 
    bindo = {
        'baik' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'sangat_baik' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'cukup' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'kurang' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'gain': 0
    }
    # bahasa inggris 
    bing = {
        'baik' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'sangat_baik' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'cukup' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'kurang' : {'tinggi':0, 'rendah':0, 'entropy':0},
        'gain': 0
    }
    for x in dtnp:
        # bing -> sangat baik 
        if x[13] == 4 and x[14] == 1:
            bing['sangat_baik']['tinggi'] += 1
        if x[13] == 4 and x[14] == 0:
            bing['sangat_baik']['rendah'] += 1
        # bing -> baik 
        if x[13] == 3 and x[14] == 1:
            bing['baik']['tinggi'] += 1
        if x[13] == 3 and x[14] == 0:
            bing['baik']['rendah'] += 1
        # bing -> cukup 
        if x[13] == 2 and x[14] == 1:
            bing['cukup']['tinggi'] += 1
        if x[13] == 2 and x[14] == 0:
            bing['cukup']['rendah'] += 1
        # bing -> kurang 
        if x[13] == 1 and x[14] == 1:
            bing['kurang']['tinggi'] += 1
        if x[13] == 1 and x[14] == 0:
            bing['kurang']['rendah'] += 1

        # bindo -> sangat baik 
        if x[12] == 4 and x[14] == 1:
            bindo['sangat_baik']['tinggi'] += 1
        if x[12] == 4 and x[14] == 0:
            bindo['sangat_baik']['rendah'] += 1
        # bindo -> baik 
        if x[12] == 3 and x[14] == 1:
            bindo['baik']['tinggi'] += 1
        if x[12] == 3 and x[14] == 0:
            bindo['baik']['rendah'] += 1
        # bindo -> cukup 
        if x[12] == 2 and x[14] == 1:
            bindo['cukup']['tinggi'] += 1
        if x[12] == 2 and x[14] == 0:
            bindo['cukup']['rendah'] += 1
        # bindo -> kurang 
        if x[12] == 1 and x[14] == 1:
            bindo['kurang']['tinggi'] += 1
        if x[12] == 1 and x[14] == 0:
            bindo['kurang']['rendah'] += 1

        # mat -> sangat baik 
        if x[11] == 4 and x[14] == 1:
            mat['sangat_baik']['tinggi'] += 1
        if x[11] == 4 and x[14] == 0:
            mat['sangat_baik']['rendah'] += 1
        # mat -> baik 
        if x[11] == 3 and x[14] == 1:
            mat['baik']['tinggi'] += 1
        if x[11] == 3 and x[14] == 0:
            mat['baik']['rendah'] += 1
        # mat -> cukup 
        if x[11] == 2 and x[14] == 1:
            mat['cukup']['tinggi'] += 1
        if x[11] == 2 and x[14] == 0:
            mat['cukup']['rendah'] += 1
        # mat -> kurang 
        if x[11] == 1 and x[14] == 1:
            mat['kurang']['tinggi'] += 1
        if x[11] == 1 and x[14] == 0:
            mat['kurang']['rendah'] += 1

        # uas -> sangat baik 
        if x[10] == 4 and x[14] == 1:
            uas['sangat_baik']['tinggi'] += 1
        if x[10] == 4 and x[14] == 0:
            uas['sangat_baik']['rendah'] += 1
        # uas -> baik 
        if x[10] == 3 and x[14] == 1:
            uas['baik']['tinggi'] += 1
        if x[10] == 3 and x[14] == 0:
            uas['baik']['rendah'] += 1
        # uas -> cukup 
        if x[10] == 2 and x[14] == 1:
            uas['cukup']['tinggi'] += 1
        if x[10] == 2 and x[14] == 0:
            uas['cukup']['rendah'] += 1
        # uas -> kurang 
        if x[10] == 1 and x[14] == 1:
            uas['kurang']['tinggi'] += 1
        if x[10] == 1 and x[14] == 0:
            uas['kurang']['rendah'] += 1

        # uts -> sangat baik 
        if x[9] == 4 and x[14] == 1:
            uts['sangat_baik']['tinggi'] += 1
        if x[9] == 4 and x[14] == 0:
            uts['sangat_baik']['rendah'] += 1
        # uts -> baik 
        if x[9] == 3 and x[14] == 1:
            uts['baik']['tinggi'] += 1
        if x[9] == 3 and x[14] == 0:
            uts['baik']['rendah'] += 1
        # uts -> cukup 
        if x[9] == 2 and x[14] == 1:
            uts['cukup']['tinggi'] += 1
        if x[9] == 2 and x[14] == 0:
            uts['cukup']['rendah'] += 1
        # uts -> kurang 
        if x[9] == 1 and x[14] == 1:
            uts['kurang']['tinggi'] += 1
        if x[9] == 1 and x[14] == 0:
            uts['kurang']['rendah'] += 1
        # praktikum -> sangat baik 
        if x[8] == 4 and x[14] == 1:
            pk['sangat_baik']['tinggi'] += 1
        if x[8] == 4 and x[14] == 0:
            pk['sangat_baik']['rendah'] += 1
        # praktikum -> baik 
        if x[8] == 3 and x[14] == 1:
            pk['baik']['tinggi'] += 1
        if x[8] == 3 and x[14] == 0:
            pk['baik']['rendah'] += 1
        # praktikum -> cukup 
        if x[8] == 2 and x[14] == 1:
            pk['cukup']['tinggi'] += 1
        if x[8] == 2 and x[14] == 0:
            pk['cukup']['rendah'] += 1
        # praktikum -> kurang 
        if x[8] == 1 and x[14] == 1:
            pk['kurang']['tinggi'] += 1
        if x[8] == 1 and x[14] == 0:
            pk['kurang']['rendah'] += 1
        # kehadiran -> sangat baik 
        if x[7] == 4 and x[14] == 1:
            kh['sangat_baik']['tinggi'] += 1
        if x[7] == 4 and x[14] == 0:
            kh['sangat_baik']['rendah'] += 1
        # kehadiran -> baik
        if x[7] == 3 and x[14] == 1:
            kh['baik']['tinggi'] += 1
        if x[7] == 3 and x[14] == 0:
            kh['baik']['rendah'] += 1
        # kehadiran -> cukup 
        if x[7] == 2 and x[14] == 1:
            kh['cukup']['tinggi'] += 1
        if x[7] == 2 and x[14] == 0:
            kh['cukup']['rendah'] += 1    
        # tugas -> sangat_baik 
        if x[6] == 4 and x[14] == 1:
            tg['sangat_baik']['tinggi'] += 1
        if x[6] == 4 and x[14] == 0:
            tg['sangat_baik']['rendah'] += 1
        # tugas -> baik 
        if x[6] == 3 and x[14] == 1:
            tg['baik']['tinggi'] += 1
        if x[6] == 3 and x[14] == 0:
            tg['baik']['rendah'] += 1
        # tugas -> cukup
        if x[6] == 2 and x[14] == 1:
            tg['cukup']['tinggi'] += 1
        if x[6] == 2 and x[14] == 0:
            tg['cukup']['rendah'] += 1
        # tugas -> kurang
        if x[6] == 1 and x[14] == 1:
            tg['kurang']['tinggi'] += 1
        if x[6] == 1 and x[14] == 0:
            tg['kurang']['rendah'] += 1
        # suasana belajar -> mendukung 
        if x[5] == 4 and x[14] == 1:
            sb['mendukung']['tinggi'] += 1
        if x[5] == 4 and x[14] == 0:
            sb['mendukung']['rendah'] += 1
        # suasana belajar -> tidak mendukung 
        if x[5] == 1 and x[14] == 1:
            sb['tidak_mendukung']['tinggi'] += 1
        if x[5] == 1 and x[14] == 0:
            sb['tidak_mendukung']['rendah'] += 1
        # media pembelajaran -> ebook 
        if x[4] == 4 and x[14] == 1:
            mp['ebook']['tinggi'] += 1
        if x[4] == 4 and x[14] == 0:
            mp['ebook']['rendah'] += 1
        # media pembelajaran -> ppt 
        if x[4] == 3 and x[14] == 1:
            mp['video']['tinggi'] += 1
        if x[4] == 3 and x[14] == 0:
            mp['video']['rendah'] += 1
        # media pembelajaran -> ppt
        if x[4] == 2 and x[14] == 1:
            mp['ppt']['tinggi'] += 1
        if x[4] == 2 and x[14] == 0:
            mp['ppt']['rendah'] += 1
        # media pembelajaran -> pdf
        if x[4] == 1 and x[14] == 1:
            mp['pdf']['tinggi'] += 1
        if x[4] == 1 and x[14] == 0:
            mp['pdf']['rendah'] += 1    
        # penyampaian materi -> serius santai 
        if x[3] == 4 and x[14] == 1:
            pm['serius_santai']['tinggi'] += 1
        if x[3] == 4 and x[14] == 0:
            pm['serius_santai']['rendah'] += 1
        # penyampaian materi -> serius 
        if x[3] == 3 and x[14] == 1:
            pm['serius']['tinggi'] += 1
        if x[3] == 3 and x[14] == 0:
            pm['serius']['rendah'] += 1
        # penyampaian materi -> santai 
        if x[3] == 2 and x[14] == 1:
            pm['santai']['tinggi'] += 1
        if x[3] == 2 and x[14] == 0:
            pm['santai']['rendah'] += 1
        # penyampaian materi -> membosankan 
        if x[3] == 1 and x[14] == 1:
            pm['membosankan']['tinggi'] += 1
        if x[3] == 1 and x[14] == 0:
            pm['membosankan']['rendah'] += 1
    # append semua nilai bing 
    bing['baik']['entropy'] = cse(bing['baik']['tinggi'], totalRecord) + cse(bing['baik']['rendah'], totalRecord)
    bing['sangat_baik']['entropy'] = cse(bing['sangat_baik']['tinggi'], totalRecord) + cse(bing['sangat_baik']['rendah'], totalRecord)
    bing['cukup']['entropy'] = cse(bing['cukup']['tinggi'], totalRecord) + cse(bing['cukup']['rendah'], totalRecord)
    bing['kurang']['entropy'] = cse(bing['kurang']['tinggi'], totalRecord) + cse(bing['kurang']['rendah'], totalRecord)
    bing['gain'] = dp['entropy'] - (((bing['baik']['tinggi'] + bing['baik']['rendah']) / totalRecord) * bing['baik']['entropy'])\
        - (((bing['sangat_baik']['tinggi'] + bing['sangat_baik']['rendah']) / totalRecord) * bing['sangat_baik']['entropy'])\
            - (((bing['cukup']['tinggi'] + bing['cukup']['rendah']) / totalRecord) * bing['cukup']['entropy'])\
                - (((bing['kurang']['tinggi'] + bing['kurang']['rendah']) / totalRecord) * bing['kurang']['entropy'])
    # append semua nilai bindo
    bindo['baik']['entropy'] = cse(bindo['baik']['tinggi'], totalRecord) + cse(bindo['baik']['rendah'], totalRecord)
    bindo['sangat_baik']['entropy'] = cse(bindo['sangat_baik']['tinggi'], totalRecord) + cse(bindo['sangat_baik']['rendah'], totalRecord)
    bindo['cukup']['entropy'] = cse(bindo['cukup']['tinggi'], totalRecord) + cse(bindo['cukup']['rendah'], totalRecord)
    bindo['kurang']['entropy'] = cse(bindo['kurang']['tinggi'], totalRecord) + cse(bindo['kurang']['rendah'], totalRecord)
    bindo['gain'] = dp['entropy'] - (((bindo['baik']['tinggi'] + bindo['baik']['rendah']) / totalRecord) * bindo['baik']['entropy'])\
        - (((bindo['sangat_baik']['tinggi'] + bindo['sangat_baik']['rendah']) / totalRecord) * bindo['sangat_baik']['entropy'])\
            - (((bindo['cukup']['tinggi'] + bindo['cukup']['rendah']) / totalRecord) * bindo['cukup']['entropy'])\
                - (((bindo['kurang']['tinggi'] + bindo['kurang']['rendah']) / totalRecord) * bindo['kurang']['entropy'])

    # append semua nilai matematika
    mat['baik']['entropy'] = cse(mat['baik']['tinggi'], totalRecord) + cse(mat['baik']['rendah'], totalRecord)
    mat['sangat_baik']['entropy'] = cse(mat['sangat_baik']['tinggi'], totalRecord) + cse(mat['sangat_baik']['rendah'], totalRecord)
    mat['cukup']['entropy'] = cse(mat['cukup']['tinggi'], totalRecord) + cse(mat['cukup']['rendah'], totalRecord)
    mat['kurang']['entropy'] = cse(mat['kurang']['tinggi'], totalRecord) + cse(mat['kurang']['rendah'], totalRecord)
    mat['gain'] = dp['entropy'] - (((mat['baik']['tinggi'] + mat['baik']['rendah']) / totalRecord) * mat['baik']['entropy'])\
        - (((mat['sangat_baik']['tinggi'] + mat['sangat_baik']['rendah']) / totalRecord) * mat['sangat_baik']['entropy'])\
            - (((mat['cukup']['tinggi'] + mat['cukup']['rendah']) / totalRecord) * mat['cukup']['entropy'])\
                - (((mat['kurang']['tinggi'] + mat['kurang']['rendah']) / totalRecord) * mat['kurang']['entropy'])
    # append semua nilai uas
    uas['baik']['entropy'] = cse(uas['baik']['tinggi'], totalRecord) + cse(uas['baik']['rendah'], totalRecord)
    uas['sangat_baik']['entropy'] = cse(uas['sangat_baik']['tinggi'], totalRecord) + cse(uas['sangat_baik']['rendah'], totalRecord)
    uas['cukup']['entropy'] = cse(uas['cukup']['tinggi'], totalRecord) + cse(uas['cukup']['rendah'], totalRecord)
    uas['kurang']['entropy'] = cse(uas['kurang']['tinggi'], totalRecord) + cse(uas['kurang']['rendah'], totalRecord)
    uas['gain'] = dp['entropy'] - (((uas['baik']['tinggi'] + uas['baik']['rendah']) / totalRecord) * uas['baik']['entropy'])\
        - (((uas['sangat_baik']['tinggi'] + uas['sangat_baik']['rendah']) / totalRecord) * uas['sangat_baik']['entropy'])\
            - (((uas['cukup']['tinggi'] + uas['cukup']['rendah']) / totalRecord) * uas['cukup']['entropy'])\
                - (((uas['kurang']['tinggi'] + uas['kurang']['rendah']) / totalRecord) * uas['kurang']['entropy'])
    # append semua nilai uts 
    uts['baik']['entropy'] = cse(uts['baik']['tinggi'], totalRecord) + cse(uts['baik']['rendah'], totalRecord)
    uts['sangat_baik']['entropy'] = cse(uts['sangat_baik']['tinggi'], totalRecord) + cse(uts['sangat_baik']['rendah'], totalRecord)
    uts['cukup']['entropy'] = cse(uts['cukup']['tinggi'], totalRecord) + cse(uts['cukup']['rendah'], totalRecord)
    uts['kurang']['entropy'] = cse(uts['kurang']['tinggi'], totalRecord) + cse(uts['kurang']['rendah'], totalRecord)
    uts['gain'] = dp['entropy'] - (((uts['baik']['tinggi'] + uts['baik']['rendah']) / totalRecord) * uts['baik']['entropy'])\
        - (((uts['sangat_baik']['tinggi'] + uts['sangat_baik']['rendah']) / totalRecord) * uts['sangat_baik']['entropy'])\
            - (((uts['cukup']['tinggi'] + uts['cukup']['rendah']) / totalRecord) * uts['cukup']['entropy'])\
                - (((uts['kurang']['tinggi'] + uts['kurang']['rendah']) / totalRecord) * uts['kurang']['entropy'])
    # append semua nilai praktikum
    pk['baik']['entropy'] = cse(pk['baik']['tinggi'], totalRecord) + cse(pk['baik']['rendah'], totalRecord)
    pk['sangat_baik']['entropy'] = cse(pk['sangat_baik']['tinggi'], totalRecord) + cse(pk['sangat_baik']['rendah'], totalRecord)
    pk['cukup']['entropy'] = cse(pk['cukup']['tinggi'], totalRecord) + cse(pk['cukup']['rendah'], totalRecord)
    pk['kurang']['entropy'] = cse(pk['kurang']['tinggi'], totalRecord) + cse(pk['kurang']['rendah'], totalRecord)
    pk['gain'] = dp['entropy'] - (((pk['baik']['tinggi'] + pk['baik']['rendah']) / totalRecord) * pk['baik']['entropy'])\
        - (((pk['sangat_baik']['tinggi'] + pk['sangat_baik']['rendah']) / totalRecord) * pk['sangat_baik']['entropy'])\
            - (((pk['cukup']['tinggi'] + pk['cukup']['rendah']) / totalRecord) * pk['cukup']['entropy'])\
                - (((pk['kurang']['tinggi'] + pk['kurang']['rendah']) / totalRecord) * pk['kurang']['entropy'])

    # append semua nilai kehadiran 
    kh['baik']['entropy'] = cse(kh['baik']['tinggi'], totalRecord) + cse(kh['baik']['rendah'], totalRecord)
    kh['sangat_baik']['entropy'] = cse(kh['sangat_baik']['tinggi'], totalRecord) + cse(kh['sangat_baik']['rendah'], totalRecord)
    kh['cukup']['entropy'] = cse(kh['cukup']['tinggi'], totalRecord) + cse(kh['cukup']['rendah'], totalRecord)
    kh['gain'] = dp['entropy'] - (((kh['baik']['tinggi'] + kh['baik']['rendah']) / totalRecord) * kh['baik']['entropy'])\
        - (((kh['sangat_baik']['tinggi'] + kh['sangat_baik']['rendah']) / totalRecord) * kh['sangat_baik']['entropy'])\
            - (((kh['cukup']['tinggi'] + kh['cukup']['rendah']) / totalRecord) * kh['cukup']['entropy'])

    # append semua nilai tugas 
    tg['baik']['entropy'] = cse(tg['baik']['tinggi'], totalRecord) + cse(tg['baik']['rendah'], totalRecord)
    tg['cukup']['entropy'] = cse(tg['cukup']['tinggi'], totalRecord) + cse(tg['cukup']['rendah'], totalRecord)
    tg['sangat_baik']['entropy'] = cse(tg['sangat_baik']['tinggi'], totalRecord) + cse(tg['sangat_baik']['rendah'], totalRecord)
    tg['kurang']['entropy'] = cse(tg['kurang']['tinggi'], totalRecord) + cse(tg['kurang']['rendah'], totalRecord)
    tg['gain'] = dp['entropy'] - (((tg['baik']['tinggi'] + tg['baik']['rendah']) / totalRecord) * tg['baik']['entropy'])\
        - (((tg['cukup']['tinggi'] + tg['cukup']['rendah']) / totalRecord) * tg['cukup']['entropy'])\
            - (((tg['sangat_baik']['tinggi'] + tg['sangat_baik']['rendah']) / totalRecord) * tg['sangat_baik']['entropy'])\
                - (((tg['kurang']['tinggi'] + tg['kurang']['rendah']) / totalRecord) * tg['kurang']['entropy'])

    # append semua nilai suasana belajar 
    sb['mendukung']['entropy'] = cse(sb['mendukung']['tinggi'], totalRecord) + cse(sb['mendukung']['rendah'], totalRecord)
    sb['tidak_mendukung']['entropy'] = cse(sb['tidak_mendukung']['tinggi'], totalRecord) + cse(sb['tidak_mendukung']['rendah'], totalRecord)
    sb['gain'] = dp['entropy'] - (((sb['mendukung']['tinggi'] + sb['mendukung']['rendah']) / totalRecord) * sb['mendukung']['entropy'])\
        - (((sb['tidak_mendukung']['tinggi'] + sb['tidak_mendukung']['rendah']) / totalRecord) * sb['tidak_mendukung']['entropy'])

    # append semua nilai penyampaian materi 
    pm['serius_santai']['entropy'] =  cse(pm['serius_santai']['tinggi'], totalRecord) + cse(pm['serius_santai']['rendah'], totalRecord)
    pm['serius']['entropy'] =  cse(pm['serius']['tinggi'], totalRecord) + cse(pm['serius']['rendah'], totalRecord)
    pm['santai']['entropy'] =  cse(pm['santai']['tinggi'], totalRecord) + cse(pm['santai']['rendah'], totalRecord)
    pm['membosankan']['entropy'] =  cse(pm['membosankan']['tinggi'], totalRecord) + cse(pm['membosankan']['rendah'], totalRecord)
    pm['gain'] = dp['entropy']-(((pm['membosankan']['tinggi'] + pm['membosankan']['rendah']) / totalRecord) * pm['membosankan']['entropy']) \
        - (((pm['serius_santai']['tinggi'] + pm['serius_santai']['rendah']) / totalRecord) * pm['serius_santai']['entropy']) \
            - (((pm['santai']['tinggi'] + pm['santai']['rendah']) / totalRecord) * pm['santai']['entropy']) \
                - (((pm['serius']['tinggi'] + pm['serius']['rendah']) / totalRecord) * pm['serius']['entropy'])

    # append semua nilai media pembelajaran
    mp['pdf']['entropy'] = cse(mp['pdf']['tinggi'], totalRecord) + cse(mp['pdf']['rendah'], totalRecord)
    mp['video']['entropy'] = cse(mp['video']['tinggi'], totalRecord) + cse(mp['video']['rendah'], totalRecord)
    mp['ppt']['entropy'] = cse(mp['ppt']['tinggi'], totalRecord) + cse(mp['ppt']['rendah'], totalRecord)
    mp['ebook']['entropy'] = cse(mp['ebook']['tinggi'], totalRecord) + cse(mp['ebook']['rendah'], totalRecord)
    mp['gain'] = dp['entropy']-(((mp['pdf']['tinggi'] + mp['pdf']['rendah']) / totalRecord) * mp['pdf']['entropy']) \
        - (((mp['video']['tinggi'] + mp['video']['rendah']) / totalRecord) * mp['video']['entropy']) \
            - (((mp['ppt']['tinggi'] + mp['ppt']['rendah']) / totalRecord) * mp['ppt']['entropy']) \
                - (((mp['ebook']['tinggi'] + mp['ebook']['rendah']) / totalRecord) * mp['ebook']['entropy'])

    dGain = []
    dGain.append(pm['gain'])
    dGain.append(mp['gain'])
    dGain.append(sb['gain'])
    dGain.append(tg['gain'])
    dGain.append(kh['gain'])
    dGain.append(pk['gain'])
    dGain.append(uas['gain'])
    dGain.append(uts['gain'])
    dGain.append(mat['gain'])
    dGain.append(bindo['gain'])
    dGain.append(bing['gain'])

    return render_template('normalisasi-data-training.html', dSiswa=dSiswa, penyampaianMateri=pm, mediaPembelajaran=mp, suasanaBelajar=sb, tugas=tg, kehadiran=kh, praktikum=pk, uts=uts, uas=uas, matematika=mat, bindo=bindo, bing=bing, dp=dp, dGain=dGain)

@app.route('/data-testing')
def data_tesing():
    dSiswaTraining = []
    dSiswaTesting = []
    dSiswaAll = []
    dataSiswaTraining = pd.read_excel("./DATA_TRAINING.xlsx")
    dataSiswaTesting = pd.read_excel("./DATA_TESTING.xlsx")

    dataSiswaTraining.replace({'penyampaian_materi':{'Serius Santai':4, 'Serius':3, 'Santai':2, 'Membosankan':1}},inplace=True)
    dataSiswaTraining.replace({'media_pembelajaran':{'Ebook':4, 'Video':3, 'PPT':2, 'PDF':1}},inplace=True)
    dataSiswaTraining.replace({'suasana_belajar':{'Mendukung':4, 'Tidak Mendukung':1}},inplace=True)
    dataSiswaTraining.replace({'tugas':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswaTraining.replace({'kehadiran':{'Sangat Baik':4, 'Baik':3, 'Cukup':2}},inplace=True)
    dataSiswaTraining.replace({'praktikum':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswaTraining.replace({'uts':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswaTraining.replace({'uas':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswaTraining.replace({'matematika':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswaTraining.replace({'bindo':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswaTraining.replace({'bing':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    # dataSiswaTraining.replace({'pemahaman':{'Tinggi':1, 'Rendah':0}},inplace=True)
    dstnp = dataSiswaTraining.to_numpy()

    dataSiswaTesting.replace({'penyampaian_materi':{'Serius Santai':4, 'Serius':3, 'Santai':2, 'Membosankan':1}},inplace=True)
    dataSiswaTesting.replace({'media_pembelajaran':{'Ebook':4, 'Video':3, 'PPT':2, 'PDF':1}},inplace=True)
    dataSiswaTesting.replace({'suasana_belajar':{'Mendukung':4, 'Tidak Mendukung':1}},inplace=True)
    dataSiswaTesting.replace({'tugas':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswaTesting.replace({'kehadiran':{'Sangat Baik':4, 'Baik':3, 'Cukup':2}},inplace=True)
    dataSiswaTesting.replace({'praktikum':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswaTesting.replace({'uts':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswaTesting.replace({'uas':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswaTesting.replace({'matematika':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswaTesting.replace({'bindo':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswaTesting.replace({'bing':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    # dataSiswaTesting.replace({'pemahaman':{'Tinggi':1, 'Rendah':0}},inplace=True)
    dtestnp = dataSiswaTesting.to_numpy()
    
    ord = 1
    for x in dtestnp:
        dSatuan = {}
        dSatuan['nama'] = x[1]
        dSatuan['ord'] = ord
        dSatuan['kelas'] = x[2]
        dSatuan['penyampaian_materi'] = x[3]
        dSatuan['media_pembelajaran'] = x[4]
        dSatuan['suasana_belajar'] = x[5]
        dSatuan['tugas'] = x[6]
        dSatuan['kehadiran'] = x[7]
        dSatuan['praktikum'] = x[8]
        dSatuan['uts' ] = x[9]
        dSatuan['uas'] = x[10]
        dSatuan['matematika'] = x[11]
        dSatuan['b_indo'] = x[12]
        dSatuan['b_inggris'] = x[13]
        dSatuan['pemahaman'] = x[14]
        dSiswaTesting.append(dSatuan)
        dSiswaAll.append(dSatuan)
        ord += 1

    ordSec = 1
    for x in dstnp:
        dSatuan = {}
        dSatuan['nama'] = x[1]
        dSatuan['ord'] = ord
        dSatuan['kelas'] = x[2]
        dSatuan['penyampaian_materi'] = x[3]
        dSatuan['media_pembelajaran'] = x[4]
        dSatuan['suasana_belajar'] = x[5]
        dSatuan['tugas'] = x[6]
        dSatuan['kehadiran'] = x[7]
        dSatuan['praktikum'] = x[8]
        dSatuan['uts' ] = x[9]
        dSatuan['uas'] = x[10]
        dSatuan['matematika'] = x[11]
        dSatuan['b_indo'] = x[12]
        dSatuan['b_inggris'] = x[13]
        dSatuan['pemahaman'] = x[14]
        dSiswaAll.append(dSatuan)
        ordSec += 1
    
    n_x = []
    n_y = []

    for x in dSiswaAll:
        # print(x)
        x_satuan = [x['penyampaian_materi'], x['media_pembelajaran'], x['suasana_belajar'], x['tugas'], x['kehadiran'], x['praktikum'], x['uts'], x['uas'], x['matematika'], x['b_indo'], x['b_inggris']]
        n_x.append(x_satuan)
        n_y.append(x['pemahaman'])

    
    X = n_x
    Y = n_y

    feature_names =  ['penyampaian_materi', 'media_pembelajaran', 'suasana_belajar','tugas','kehadiran','praktikum', 'uts','uas','matematika','bindo', 'bing']
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)

    benar = 0
    salah = 0
    ord = 0
    for x in dtestnp:
        if x[12] == dstnp[ord][12] or x[13] == dstnp[ord][13]:
            benar += 1
        ord += 1


    r = export_text(clf, feature_names=feature_names)
    file = open('static/log.txt', 'w')
    print(r, file = file)    
    file.close()
    
    tDataTraining = 0
    tDataTest = 0

    for x in dstnp:
        tDataTraining += 1

    for x in dtestnp:
        tDataTest += 1
    salah = tDataTest - benar

    akurasi = (benar / tDataTest) * 100

    return render_template('data-testing.html',akurasi= akurasi, dSiswa=dSiswaTesting, tDataTraining=tDataTraining, tDataTest=tDataTest, benar=benar, salah=salah)

@app.route('/prediksi')
def prediksi():
    dr = {'BASE_URL' : BASE_URL}
    sLogin = cekLogin()
    if sLogin == "no":
        stLogin = {'stLogin' : 'no'}
        return render_template('home.html', dRes=dr, stLogin=stLogin)
    else:
        return render_template('prediksi.html')

@app.route('/proses-prediksi', methods=('GET', 'POST'))
def proses_prediksi():
    nama = request.form['txtNamaSiswa']
    pms = request.form['txtPm']
    mps = request.form['txtMp']
    pks = request.form['txtPk']
    uass = request.form['txtUas']
    mats = request.form['txtMat']
    sbs = request.form['txtSb']
    khs = request.form['txtKh']
    utss = request.form['txtUts']
    bindos = request.form['txtBindo']
    bings = request.form['txtBing']
    tgs = request.form['txtTg']

    dSatuan = {}
    dSatuan['nama'] = nama
    dSatuan['kelas'] = ''
    dSatuan['penyampaian_materi'] = pms
    dSatuan['media_pembelajaran'] = mps
    dSatuan['suasana_belajar'] = sbs
    dSatuan['tugas'] = tgs
    dSatuan['kehadiran'] = khs
    dSatuan['praktikum'] = pks
    dSatuan['uts' ] = utss
    dSatuan['uas'] = uass
    dSatuan['matematika'] = mats
    dSatuan['b_indo'] = bindos
    dSatuan['b_inggris'] = bings

    inPub = random.randint(10, 50)

    dSiswaTraining = []
    dSiswaAll = []
    dataSiswaTraining = pd.read_excel("./DATA_TRAINING.xlsx")

    dataSiswaTraining.replace({'penyampaian_materi':{'Serius Santai':4, 'Serius':3, 'Santai':2, 'Membosankan':1}},inplace=True)
    dataSiswaTraining.replace({'media_pembelajaran':{'Ebook':4, 'Video':3, 'PPT':2, 'PDF':1}},inplace=True)
    dataSiswaTraining.replace({'suasana_belajar':{'Mendukung':4, 'Tidak Mendukung':1}},inplace=True)
    dataSiswaTraining.replace({'tugas':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswaTraining.replace({'kehadiran':{'Sangat Baik':4, 'Baik':3, 'Cukup':2}},inplace=True)
    dataSiswaTraining.replace({'praktikum':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswaTraining.replace({'uts':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswaTraining.replace({'uas':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswaTraining.replace({'matematika':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswaTraining.replace({'bindo':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    dataSiswaTraining.replace({'bing':{'Sangat Baik':4, 'Baik':3, 'Cukup':2, 'Kurang':1}},inplace=True)
    # dataSiswaTraining.replace({'pemahaman':{'Tinggi':1, 'Rendah':0}},inplace=True)
    dstnp = dataSiswaTraining.to_numpy()

    ord = 0
    status = ""
    nFaktor = 0
    for n in range(inPub):
        # print(str(utss) + " - " + str(dstnp[ord][9]))
        if int(utss) == dstnp[ord][9] or int(uass) == dstnp[ord][10]:
            nFaktor += 1
        ord += 1

    per2 = inPub / 2
    print(nFaktor)
    if nFaktor > per2:
        status = "Tinggi"
    else:
        status = "Rendah"

    return render_template('hasil-prediksi.html', status=status)

def getUserJson():
    with open('login_user.json', 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
    
    return json_object

def cekLogin():
    with open('auth_status.json', 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)

    return json_object['status_login']

def cse(nKriteria, totalData):
    if nKriteria == 0:
        nKriteria = 1
    
    result = -(nKriteria / totalData) * (log(nKriteria / totalData) / log(2))
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
