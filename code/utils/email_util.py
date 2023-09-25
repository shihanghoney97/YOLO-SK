import email
import smtplib
from email import message
from email.mime.text import MIMEText
import time
import os
from pathlib import Path


USER='xxxxxxxxxxx'
PASSWD='xxxxxxxxxxxxxxxxx'


class Email:
    def __init__(self) -> None:
          self.smtp=smtplib.SMTP_SSL("smtp.163.com",994)
          self.smtp.login(user=USER,password=PASSWD)


    def send_email(self,subject='服务器训练任务已经结束',content='服务器训练任务已经结束',recver=['18237010193@qq.com','1147455384@qq.com']):
        """
        主题  内容  接收人
        """
        message=MIMEText(content,'plain','utf-8')
        message['Subject']=subject
        message['To']=",".join(recver)
        message['From']=USER
        self.smtp.sendmail(from_addr=USER,to_addrs=recver,msg=message.as_string())
        self.smtp.close


    @staticmethod
    def get_last_log(path='/home/sh/sh_code/yolov5_GIoU_lq/logs/'):
        path=Path(path)
        log_list=os.listdir(path)
        log_list.sort(key=lambda x:os.path.getmtime(path/x))
        with open(path/log_list[-1],'r') as fp:
            content_list=fp.readlines()
            map5,map95=content_list[-32:-31][0].strip().split()[-2:]
            return map5,map95

if __name__ == '__main__':
    stamp=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
    print(stamp)
    email=Email()
    map5,map95=email.get_last_log()
    email.send_email(subject=f'训练结束,mAP@.5={map5},完成时间{stamp}',content=f'老婆，你的训练任务结束啦，快去看看结果吧-{stamp}\n\nmAP@.5={map5}')
