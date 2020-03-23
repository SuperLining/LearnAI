import cv2
import pyzbar.pyzbar as pyzbar
import pymysql
import datetime

# 定义标志，用来结束程序运行
flag = True
# 将二维码信息存入到MySQL数据库
def insert(bid, isbn, bname,sno,times):
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
    sql = "insert into book_in(bid,isbn,bname,sno,times) values(%s,%s,%s,%s,%s)"
    try:
        # 执行sql语句
        if cursor.execute(sql, (bid,isbn,bname,sno,times)) != -1:
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
def decode(image):
    # 解析图像
    inbook = []
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
        bid = data[0]
        # 将信息存入数据库
        # 绘出图像上二维码类型和二维码的数据
        text = "{} ".format(barcodeType, barcodeData)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 125), 2)
        # 向终端打印条形码数据和条形码类型
        print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
        config = {'host':'127.0.0.1', 'port':3306, 'user':'root','password':'123456', 'db':'pydata','charset':'utf8mb4','cursorclass':pymysql.cursors.DictCursor,}
        con= pymysql.connect(**config)
        try:
            with con.cursor() as cursor:
                sql="select * from book_out where bid=bid"
                cursor.execute(sql)
                result=cursor.fetchall()
        finally:
            con.close()
        df2=result
        print('在借书表里有:',df2)
        times = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for z in df2:
            e = z['bid']
            a = z['isbn']
            b = z['bname']
            c = z['sno']
            inbook.append(e)
            inbook.append(a)
            inbook.append(b)
            inbook.append(c)
        insert(inbook[0],inbook[1],inbook[2],inbook[3],times)

        config = {'host':'127.0.0.1', 'port':3306, 'user':'root','password':'123456', 'db':'pydata','charset':'utf8mb4','cursorclass':pymysql.cursors.DictCursor,}
        con= pymysql.connect(**config)
        try:
            with con.cursor() as cursor:
                sql2="delete from book_out where bid=bid"
                cursor.execute(sql2)
                con.commit()
                print("delete OK")
        except:
                con.rollback()
        con.close()
    return image

# 调用摄像头识别二维码
def detect():
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
        img = decode(gray)
        # 按Q键退出程序
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # 显示图像
        cv2.imshow("camera", img)
    # 释放摄像头资源
    cap.release()
    # 关闭显示图像的窗口
    cv2.destroyAllWindows()

# 主函数入口
if __name__ == '__main__':
    detect()
