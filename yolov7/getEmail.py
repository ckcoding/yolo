import requests
import time
import json
import re
domain_url = 'https://tempmail.plus/api/'
#取当前时间戳的后3位
current_timestamp = str(int(time.time() * 1000))[-3:]

def get_email(email, limit="20", epin=''):
  email_url = domain_url + 'mails?email=' + email + '&limit=' + str(limit) + '&epin=' + epin
  response = requests.get(email_url)
  return response.json()

def get_email_body(email_id, email, epin=''):
  body_url = domain_url + 'mails/' + str(email_id) + '?email=' + email + '&epin=' + epin
  response = requests.get(body_url)
  return response.json()

def random_email():
  email = "mo" + str(current_timestamp) + "@fexbox.org"
  return email

def send_email(random_email):
  send_email_url = "https://px.xinyo.me/api?scheme=passport/comm/sendEmailVerify"
  data = {
    "email": random_email,
    "recaptcha_data": ""
  }
  headers = {
    'Content-Type': 'application/json'
  }
  response = requests.post(
    send_email_url,
    json=data,
    headers=headers
  )
  return response.json()

def get_verification_code(email_body):
    pattern = r'验证码是：(\d{6})'
    if email_body.get('html'):
        match = re.search(pattern, email_body['html'])
        if match:
            return match.group(1)
    return None
# 注册
def register(email, verification_code):
  register_url = "https://px.xinyo.me/api?scheme=passport/auth/register"
  data = {
    "email": email,
    "password": email,
    "email_code": verification_code,
    "invite_code": "",
    "recaptcha_data": ""
  }
  headers = {
    'Content-Type': 'application/json'
  }
  response = requests.post(
    register_url,
    json=data,
    headers=headers
  )
  return response.json()

#https://px.xinyo.me/api?scheme=user/getSubscribe
def get_subscribe(token):
  subscribe_url = "https://px.xinyo.me/api?scheme=user/getSubscribe"
  headers = {
    'Authorization': token,
    'referer': 'https://px.xinyo.me/dashboard'
  }
  response = requests.get(subscribe_url, headers=headers)
  return response.json()

if __name__ == '__main__':
  random_email = random_email()
  print('随机邮箱是:', random_email)
  # random_email = 'random9977@mailto.plus'
  send_email = send_email(random_email)
  if send_email['success'] == True:
    print('邮件发送成功，等待15秒')
    time.sleep(15)
    email_arr = get_email(random_email).get('mail_list', [])
    # print(email_arr)
    if len(email_arr) > 0:
      print('邮件获取成功')
      email_id = email_arr[0]['mail_id']
      email_body = get_email_body(email_id, random_email)
      verification_code = get_verification_code(email_body)
      print(f"验证码是: {verification_code}")
      register = register(random_email, verification_code)
      # print(register)
      if register['success'] == True:
        token = register['data']['auth_data']
        subscribe = get_subscribe(token)
        print('注册成功')
        print('订阅地址是:', subscribe['data']['subscribe_url'])
      else:
        print('注册失败')
        print(register)
    
