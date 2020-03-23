# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 11:43:47 2018

@author: Administrator
"""

'''
调用opencv的库实现人脸识别
'''
import pyzbar.pyzbar as pyzbar
import cv2
import numpy as np
import os
import shutil
import pymysql


def insert(sno,sname,face,id):
    # 定义全局变量
    global flag
    # 创建数据库连接对象
    conn = pymysql.connect(host="127.0.0.1",user="root",password="123456",db="pydata",port=3306,charset="utf8")
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = conn.cursor()
    # SQL插入语句
    sql = "insert into stu(sno,sname,face,id) values (%s,%s,%s,%s)"
    try:
        # 执行sql语句
        if cursor.execute(sql,(sno,sname,face,id)) != -1:
            flag = False
        # 提交到数据库执行
        conn.commit()
    except Exception as e:
        # 如果发生错误则回滚并打印错误信息
        conn.rollback()
        print(e)
    # 关闭游标
    cursor.close()
    # 关闭数据库连接
    conn.close()


#采集自己的人脸数据
def generator(data,id):
    '''
    打开摄像头，读取帧，检测该帧图像中的人脸，并进行剪切、缩放
    生成图片满足以下格式：
    1.灰度图，后缀为 .png
    2.图像大小相同
    params:
        data:指定生成的人脸数据的保存路径
    '''
    sname=input('姓名:')
    sno = input('学号:')

    #如果路径存在则删除路径
    path=os.path.join(data,sname)
    if os.path.isdir(path):
        shutil.rmtree(path)
    #创建文件夹
    os.mkdir(path)
    print(path)
    #创建一个级联分类器
    face_casecade=cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    #打开摄像头
    # video="http://admin:admin@192.168.43.1:8081/"
    camera=cv2.VideoCapture(0)
    cv2.namedWindow('Dynamic')
    #计数
    count=0
    while(True):
        #读取一帧图像
            ret, frame = camera.read()
            gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_casecade.detectMultiScale(gray_img, 1.3, 5)
            for (x,y,w,h) in faces:
                #在原图上绘制矩形
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                #调整图像大小
                new_frame=cv2.resize(gray_img[y:y+h,x:x+w],(92,112))
                #保存人脸i
                cv2.imwrite('%s/%s.jpg'% (path, str(count)),new_frame)
                # print(count)
                if count<10:
                    count += 1
            cv2.imshow('Dynamic',frame)
            #按下q键退出
            if cv2.waitKey(1000//12) & 0xff==ord('q'):
                break
    camera.release()
    cv2.destroyAllWindows()
    face = path
    lining = []
    lining.append(sno)
    lining.append(sname)
    lining.append(face)
    lining.append(id)
    sno = lining[0]
    sname = lining[1]
    face = lining[2]
    id = lining[3]
    insert(sno,sname,face,id)


#载入图像   读取ORL人脸数据库，准备训练数据
def LoadImages():
    '''1
    加载图片数据用于训练
    params:
        data:训练数据所在的目录，要求图片尺寸一样
    ret:
        images:[m,height,width]  m为样本数，height为高，width为宽
        names：名字的集合
        labels：标签
    '''
    images=[]
    name = []

    lable=0
    config = {
          'host':'127.0.0.1',
          'port':3306,#MySQL默认端口
          'user':'root',#mysql默认用户名
          'password':'123456',
          'db':'pydata',#数据库
          'charset':'utf8mb4',
          'cursorclass':pymysql.cursors.DictCursor,
          }

# 创建连接
    con= pymysql.connect(**config)
# 执行sql语句
    try:
        with con.cursor() as cursor:
            sql="select face,sname from stu"
            cursor.execute(sql)
            result=cursor.fetchall()
    finally:
        con.close()
    df=result
    #print(df)
    list_name = []
    lab=[]
    for x in df:
        x = x['face']
        list_name.append(x)

    for z in df:
        z = z['sname']
        name.append(z)

    for i in list_name:
        data = i
        # print(data)
        for filename in os.listdir(data):
            imgpath=os.path.join(data,filename)
            img=cv2.imread(imgpath,cv2.IMREAD_COLOR)
            gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            images.append(gray_img)
            lab.append(lable)
        lable+=1
    # print(lab)
    images=np.asarray(images)
    # print(images)
    labels=np.asarray(lab)
    # print(labels)
    names = np.asarray(name)
    # print(names)
    # print('图像训练结束!')
    # print(labels)
    return images,labels,names

#检验训练结果
def FaceRec():
    #加载训练的数据
    X,y,names=LoadImages()
    # print('x',X)
    # print('y',y)
    # print('names',names)
    model = cv2.face.EigenFaceRecognizer_create()
    model.train(X,y)
    #打开摄像头
    camera=cv2.VideoCapture(0)
    cv2.namedWindow('Dynamic')

    #创建级联分类器
    face_casecade=cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

    while(True):
        #读取一帧图像
        #ret:图像是否读取成功
        #frame：该帧图像
        ret,frame=camera.read()
        #判断图像是否读取成功
        #print('ret',ret)
        if ret:
            #转换为灰度图
            gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            #利用级联分类器鉴别人脸
            faces=face_casecade.detectMultiScale(gray_img,1.3,5)

            #遍历每一帧图像，画出矩形
            for (x,y,w,h) in faces:
                frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)  #蓝色
                roi_gray=gray_img[y:y+h,x:x+w]
                try:
                    #将图像转换为宽92 高112的图像
                    #resize（原图像，目标大小，（插值方法）interpolation=，）
                    roi_gray=cv2.resize(roi_gray,(92,112),interpolation=cv2.INTER_LINEAR)
                    params=model.predict(roi_gray)

                    print('Label:%s,confidence:%.2f'%(params[0],params[1]))
                    '''
                    putText:给照片添加文字
                    putText(输入图像，'所需添加的文字'，左上角的坐标，字体，字体大小，颜色，字体粗细)
                    '''
                    cv2.putText(frame,names[params[0]],(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)

                except:
                    continue
            cv2.imshow('Dynamic',frame)
                #按下q键退出
            if cv2.waitKey(100) & 0xff==ord('q'):
                break

    camera.release()
    cv2.destroyAllWindows()
    config = {'host':'127.0.0.1','port':3306, 'user':'root','password':'123456', 'db':'pydata','charset':'utf8mb4','cursorclass':pymysql.cursors.DictCursor,}
    con= pymysql.connect(**config)
    try:
        with con.cursor() as cursor:
            sql="select sno,sname from stu where id='params[0]'"
            cursor.execute(sql)
            result=cursor.fetchall()
    finally:
        con.close()
    df=result
    print(df)
    book=[]
    for x in df:
        x = x['sno']
        book.append(x)
    # print(book)
    sno2=book[0]
    return sno2

#查询借书情况
def book_out(sno):
    config = {'host':'127.0.0.1', 'port':3306, 'user':'root','password':'123456', 'db':'pydata','charset':'utf8mb4','cursorclass':pymysql.cursors.DictCursor,}
    con2= pymysql.connect(**config)
    try:
        with con2.cursor() as cursor:
            sql2="select bname from book_out where sno=sno"
            cursor.execute(sql2)
            result2=cursor.fetchall()
    finally:
        con2.close()
    df2=result2
    print(df2)

flag = True
# 将二维码信息存入到MySQL数据库
def pick(bid, isbn, bname,sno):

    # 定义全局变量
    global flag
    # 创建数据库连接对象
    conn = pymysql.connect(
        # 数据库的IP地址
        host="127.0.0.1",
        # 数据库用户名称
        user="root",
        # 数据库用户密码
        password="123456",
        # 数据库名称
        db="pydata",
        # 数据库端口名称
        port=3306,
        # 数据库的编码方式 注意是utf8
        charset="utf8"
    )

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = conn.cursor()

    # SQL插入语句
    sql = "insert into book_out(bid,isbn,bname,sno) values(%s,%s,%s,%s)"
    try:
        # 执行sql语句
        if cursor.execute(sql, (bid,isbn,bname,sno)) != -1:
            flag = False
        # 提交到数据库执行
        conn.commit()
    except Exception as e:
        # 如果发生错误则回滚并打印错误信息
        conn.rollback()
        print(e)
    # 关闭游标
    cursor.close()
    # 关闭数据库连接
    conn.close()

# 解析二维码
def decode(image,sno):
    # 解析图像
    barcodes = pyzbar.decode(image)

    for barcode in barcodes:
        # 提取二维码的边界框的位置
        (x, y, w, h) = barcode.rect
        # 画出图像中条形码的边界框
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 二维码类型
        barcodeType = barcode.type
        # 二维码数据
        barcodeData = barcode.data.decode("utf-8")
        data  = barcodeData.split(' ')

        data.append(sno)
        bid =  data[0]
        isbn = data[1]
        bname = data[2]
        sno = data[3]
        # 将信息存入数据库
        #insert(barcodeType, barcodeData)
        pick(bid,isbn,bname,sno)
        # 绘出图像上二维码类型和二维码的数据
        text = "{} ".format(barcodeType, barcodeData)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 125), 2)

        # 向终端打印条形码数据和条形码类型
        print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
    return image


# 调用摄像头识别二维码
def detect(sno2):
    # 调用内置摄像头
    #camera = cv2.VideoCapture(0) # 参数0代表内置摄像头，参数1代表外置摄像头
    video="http://admin:admin@192.168.43.1:8081/"   #此处@后的ipv4 地址需要修改为自己的地址
    cap =cv2.VideoCapture(video)
    # 实时显示图像
    while flag:
        # 读取当前帧
        ret, frame = cap.read()  # ret:boolean值,表示是否正常打开摄像头  frame:获取当前帧图像
        # 转为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 调用解析函数
        img = decode(gray,sno2)
        # 按Q键退出程序
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # 显示图像
        cv2.imshow("camera", img)
    # 释放摄像头资源
    cap.release()
    # 关闭显示图像的窗口
    cv2.destroyAllWindows()


if __name__=='__main__':
    i=int(input('人数:'))
    for k in range(3,i):
        data='data'
        generator(data,i)
    #LoadImages()

    # sno2=FaceRec()
    #查询借书情况
    #book_out(sno2)
    #进行借书
    # detect(sno2)
