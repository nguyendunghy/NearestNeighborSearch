import requests as re

if __name__ == '__main__':
    response = re.post('http://127.0.0.1:8080/predict',
                       json={'list_text': ['hello', 'I love cats']})
    print(response.text)
