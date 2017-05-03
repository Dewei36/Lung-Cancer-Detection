import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    file_id1 = '0Bz-jINrxV740QTZNenZadmw4NEE'
    destination1 = 'subset1.zip'
    
    file_id2 = '0Bz-jINrxV740OFRGZ0lvZ1pZU28'
    destination2 = 'subset2.zip'
    
    file_id3 = '0Bz-jINrxV740Z0NoSW9JRE9RRVk'
    destination3 = 'subset3.zip'
    
    file_id4 = '0Bz-jINrxV740SDB3R0hYalJGa2M'
    destination4 = 'subset4.zip'
    
    file_id5 = '0Bz-jINrxV740Nm1fQWhTZUVia2s'
    destination5 = 'subset5.zip'
    
    file_id6 = '0Bz-jINrxV740YXBpYV9MQkVxSk0'
    destination6 = 'subset6.zip'
    
    file_id7 = '0Bz-jINrxV740YTM0X2dQcGZ1eFU'
    destination7 = 'subset7.zip'
    
    file_id8 = '0Bz-jINrxV740WEpXakYyRWxaSm8'
    destination8 = 'subset8.zip'
    
    file_id9 = '0Bz-jINrxV740YVZFMEdpQmZrOTQ'
    destination9 = 'subset9.zip'
    
    download_file_from_google_drive(file_id1, destination1)
    download_file_from_google_drive(file_id2, destination2)
    download_file_from_google_drive(file_id3, destination3)
    download_file_from_google_drive(file_id4, destination4)
    download_file_from_google_drive(file_id5, destination5)
    download_file_from_google_drive(file_id6, destination6)
    download_file_from_google_drive(file_id7, destination7)
    download_file_from_google_drive(file_id8, destination8)
    download_file_from_google_drive(file_id9, destination9)