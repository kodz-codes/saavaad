#Dependencies
import pytesseract
import json 
import os
import torch
import io
import cv2
import pyttsx3
import numpy as np
import PyPDF2 
import bcrypt
from PIL import Image
from transformers import T5Config
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import MarianMTModel, MarianTokenizer, T5Tokenizer, T5ForConditionalGeneration, T5Config
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from kraken import pageseg
from kraken import binarization
from pymongo import MongoClient
from bson import ObjectId

#Your API Definition
app = Flask(__name__)
CORS(app)

client = MongoClient('mongodb://localhost:27017/')
db = client['saavaad']

print(db.list_collection_names())
if 'data' not in db.list_collection_names():
    user_collection_data = db.create_collection('data')
else:
    user_collection_data = db['data']

if 'users' not in db.list_collection_names():
    user_collection_login = db.create_collection('users')
else:
    user_collection_login = db['users']


#MongoDB Login system 
@app.route('/signup', methods=['POST'])
def signup():
    username = request.json['username']
    password = request.json['password'].encode('utf-8')
    email = request.json['email']
    hashed_password = bcrypt.hashpw(password, bcrypt.gensalt()).decode('utf-8')
    user = user_collection_login.find_one({'username': username})
    if user:
        print('i am in singup fail')
        return jsonify('fail')
    else:
        print('i am in singup success')
        new_user = {
            'username': username,
            'email': email,
            'password': hashed_password
        }
        result = user_collection_login.insert_one(new_user)
    return jsonify('Hello '+new_user['username']+' Welcome Saavaad, your profile has been created!')
    
@app.route('/login', methods=['POST'])
def login():
    username = request.json['username']
    password = request.json['password'].encode('utf-8')
    print(username)
    print(password)
    user = user_collection_login.find_one({'username': username})
    if user and bcrypt.checkpw(password, user['password'].encode('utf-8')):
         print('i am in login success')
         print(user['username'])
         return jsonify(user['username'])
    else:
        print('i am in login fail')
        return jsonify('fail')

@app.route('/forgot_password_username', methods = ['POST'])
def forgot_password_username():
    username = request.json['username']
    user = user_collection_login.find_one({'username': username})
    if user:
        return jsonify('User Exists')
    else:
        return jsonify('fail')

@app.route('/forgot_password', methods = ['POST'])
def forgot_password():
    username = request.json['username']
    change_password = request.json['change_password'].encode('utf-8')
    encryp_change_password = bcrypt.hashpw(change_password, bcrypt.gensalt()).decode('utf-8')
    user = user_collection_login.find_one({'username': username})
    if user:
        print(f"Found user {user['_id']} with username {username}")
        result = user_collection_login.update_one({'username': username}, {'$set': {'password': encryp_change_password}})
        if result.modified_count == 1:
            print('modified')
        return jsonify('Password updated successfully!')
    else:
        return jsonify('Error!')


@app.route('/delete_account', methods=['POST'])
def delete_account():
    # Get the username of the user to delete from the request data
    username = request.json['username']
    email = request.json['email']  
    # Find the user and delete their account
    result = user_collection_login.find_one_and_delete({'username': username, 'email':email})
    print(result)
    if result:
        # If a user was deleted successfully
        return jsonify({'status': 'success', 'message': 'User account deleted successfully!'})
    else:
        # If no user was found with the given username
        return jsonify({'status': 'fail', 'message': 'No user found with the given username and email!'})



#Processing through models
@app.route('/index', methods=['GET'])
def index():
  output = 'Hello World!'
  return jsonify(output)
  
# Defining an endpoint for prediction of summary
@app.route('/text_summarize', methods=['POST'])
def text_summarize():
        # Check-point for requests
        print("request received")

        # Loading the model
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        
        # Get the text input from the request body
        data = request.get_json()
        text = data['text']
        print(text)
        
        # Sending the text input to the ML model for processing and summarizing
        tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=10000000)
        device = torch.device('cpu')
        preprocess_text = text.strip().replace("\n","")
        t5_prepared_Text = "summarize: "+preprocess_text
        tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

        summary_ids = model.generate(tokenized_text, num_beams=4, no_repeat_ngram_size=2, min_length=30, max_length=9024, early_stopping=False)
        
        # Storing the output
        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(output)
        return jsonify(output)

# @app.route('/image_text', methods=['POST'])
# def image_text():
#     print('request received1')
#     data = request.get_json()
#     print('request received2')
#     image = data['image']
#     print('request received3')
#     img_data = base64.b64decode(image)
#     print('request received4')
#     img = Image.open(io.BytesIO(img_data))
#     print('request received5')
#     text = pytesseract.image_to_string(img)
#     return jsonify({'text': text})

@app.route('/image_text', methods=['POST'])
def image_text():
    print('request received1')
    #image = request.json['image']
    image = request.json.get('image', None)
    print('request received2')
    img_data = io.BytesIO(base64.b64decode(image))
    print('request received3')
    img = Image.open(img_data)
    print('request received4')
    text = pytesseract.image_to_string(img)
    return jsonify({'text': text})

# #Defining an endpoint for recognition of text and summarization from an image
# @app.route('/image_text', methods=['POST'])
# def image_text():
#         # if request.method == 'OPTIONS':
#         #     response = jsonify({'status': 'success'})
#         #     response.headers.add('Access-Control-Allow-Origin', '*')
#         #     response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
#         #     response.headers.add('Access-Control-Allow-Methods', 'POST')
#         #     return response
#         print("request received")
        
#         pytesseract.pytesseract.tesseract_cmd = "C:/Users/Bsc3/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"
#         #data = request.get_json()
#         #print(data)
#         # Read the uploaded file as an image
#         uploaded_file = request.files['file'].read()
#         img = Image.open(io.BytesIO(uploaded_file))
#         # Convert the PIL image to a NumPy array
#         img_arr = np.array(img)
#         # Process the image using pytesseract
#         text = pytesseract.image_to_string(img_arr)   
#         print(text)
#         return jsonify(text)

#Defining an endpoint for recognition of text and summarization from an image
@app.route('/image_text_summarize', methods=['POST'])
def image_text_summarize():
        if request.method == 'OPTIONS':
            response = jsonify({'status': 'success'})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Access-Control-Allow-Methods', 'POST')
            return response
        print("request received")
        pytesseract.pytesseract.tesseract_cmd = "C:/Users/Bsc3/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"
        # Read the uploaded file as an image
        uploaded_file = request.files['file']   
        filename = secure_filename(uploaded_file.filename)
        uploaded_file.save(filename)  # save the uploaded file
        img = cv2.imread(filename)
        # Preprocess the image (e.g., apply grayscale and thresholding)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # Pass the preprocessed image to the Tesseract OCR engine for text recognition
        text = pytesseract.image_to_string(Image.fromarray(gray), lang='eng', config='--psm 6')
        # Print the recognized text
        print(text)
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
         # Send the text input to the ML model for processing
        # Replace this with code that sends the text input to your ML model
        tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=1000000)
        device = torch.device('cpu')
        preprocess_text = text.strip().replace("\n","")
        t5_prepared_Text = "summarize: "+preprocess_text
        tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
        # summmarize 
        summary_ids = model.generate(tokenized_text, num_beams=4, no_repeat_ngram_size=2, min_length=30, max_length=1024, early_stopping=False)
        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        #output.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin', '*'))
        return jsonify(output)

#Defining an endpoint for recognition of hand-written-text and summarization from an image
@app.route('/hw_image_text_summarize', methods=['POST'])
def hw_image_text_summarize():
        if request.method == 'OPTIONS':
            response = jsonify({'status': 'success'})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Access-Control-Allow-Methods', 'POST')
            return response
        print("request received")
        device = torch.device('cpu')

        #Loading the model and the processor
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

        #load the image being uploaded by the user
        img_path = 'try1.png'
        image = cv2.imread(img_path, 0)
        img = Image.fromarray(image)

        # Binarize the image
        bw = binarization.nlbin(img)

        # Segment the image into lines
        lines = pageseg.segment(bw)

        #initializing a final_text variable to recognize the entire text from the image
        final_text = ""

        # Extraction process of text from the image
        for box in lines['boxes']:
            # Extract region from binary image
            left, top , right , bottom = box

            #print("for each box in line: ")
            #print(f"Left: {left}, Top: {top}, Right: {right}, Bottom: {bottom}")
            
            #croping a single line from the image
            region_image = bw.crop((left, top, right, bottom))
            region_image = region_image.convert("RGB")

            # Sending the image to the processor for conversion into text
            pixel_values = processor(images=region_image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)

            #decoding the generated_ids of the image
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            #saving the text from each line
            final_text = final_text + generated_text

        # Sending the text for summarization process
        # Load the T5 summarization model
        model = T5ForConditionalGeneration.from_pretrained('t5-base')

        # Send the text input to the ML model for processing
        tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=1000000)
        device = torch.device('cpu')
        preprocess_text = final_text.strip().replace("\n","")
        t5_prepared_Text = "summarize: "+preprocess_text
        tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
        
        # summmarize 
        summary_ids = model.generate(tokenized_text, num_beams=4, no_repeat_ngram_size=2, min_length=30, max_length=1024, early_stopping=False)
        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        #output.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin', '*'))
        return jsonify(output)

#Defining an endpoint for recognition of text and summarization from a PDF
@app.route('/printed_pdf_text_summarize', methods=['POST'])
def printed_pdf_text_summarize():
        print('request received')
        #data = request.get_json()
        #print(data)
        file = request.files['file']
        if 'file' not in request.files:
            return jsonify({'error': 'File Not Being Uploaded Properly'}), 400
        
      
        if file.filename == '':
            return jsonify({'error': 'File Not Being Uploaded Properly'}), 400

        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
 
        # printing number of pages in pdf file
        print("No. of pages in the pdf is : ")
        print(num_pages)

        #intialize a variable to maintain the summary of all the pages
        sum_text =""

        # getting a specific page from the pdf file and summarizing each page
        for i in range(num_pages):
            page = pdf_reader.pages[i]

            #extracting etxt from each page
            text = page.extract_text()

            #summarizing the text of each page
            model = T5ForConditionalGeneration.from_pretrained('t5-base')
            # Send the text input to the ML model for processing
            tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=100024)
            device = torch.device('cpu')
            preprocess_text = text.strip().replace("\n","")
            t5_prepared_Text = "summarize: "+preprocess_text
            tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
        
            # summmarize 
            summary_ids = model.generate(tokenized_text, num_beams=4, no_repeat_ngram_size=2, min_length=30, max_length=1024, early_stopping=False)
            output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
            #adding all pages summary
            sum_text = sum_text + output

        #printing the summarized text
        print(sum_text)
        return jsonify(sum_text)      

@app.route('/translate_en_en', methods=['POST'])
def translate_en_en():
    print("request received")
    data = request.get_json()
    input_text = data['text']
    print(input_text)
    final_text = input_text
    #returning the text
    return jsonify(final_text)

@app.route('/translate_en_hi', methods=['POST'])
def translate_en_hi():
    print("request received")
     #Loading ml model for English to Hindi translation
    model_name = 'Helsinki-NLP/opus-mt-en-hi'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    #receive text from the user
    data = request.get_json()
    input_text = data['text']
    print(input_text)

    #making translation using ml model
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids, max_new_tokens=10000)
    final_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(final_text)
    #returning the text
    return jsonify(final_text)

@app.route('/translate_hi_en', methods=['POST'])
def translate_hi_en():
    print("request received")
    #Loading ml model for Hindi to English translation
    model_name = 'Helsinki-NLP/opus-mt-hi-en' 
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    #receive text from the user
    data = request.get_json()
    input_text = data['text']

    #making translation using ml model
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids, max_new_tokens=10000)
    final_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #returning the text
    return jsonify(final_text)

@app.route('/translate_bn_en', methods=['POST'])
def translate_bn_en():
    print('request received')
    #Loading ml model for Bengali to English translation
    model_name = 'Helsinki-NLP/opus-mt-bn-en' 
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    #receive text from the user
    data = request.get_json()
    input_text = data['text']

    #making translation using ml model
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids, max_new_tokens=10000)
    final_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #returning the text
    return jsonify(final_text)

@app.route('/translate_ml_en', methods=['POST'])
def translate_ml_en():
    print('request received')
    #Loading ml model for Malyalam to English translation
    model_name = 'Helsinki-NLP/opus-mt-ml-en' 
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    #receive text from the user
    data = request.get_json()
    input_text = data['text']

    #making translation using ml model
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids, max_new_tokens=10000)
    final_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #returning the text
    return jsonify(final_text)

@app.route('/translate_en_ml', methods=['POST'])
def translate_en_ml():
    print('request received')
    #Loading ml model for Malyalam to English translation
    model_name = 'Helsinki-NLP/opus-mt-en-ml' 
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    #receive text from the user
    data = request.get_json()
    input_text = data['text']

    #making translation using ml model
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids, max_new_tokens=10000)
    final_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #returning the text
    return jsonify(final_text)

@app.route('/translate_en_mr', methods=['POST'])
def translate_en_mr():
    print('request received')
    #Loading ml model for Malyalam to English translation
    model_name = 'Helsinki-NLP/opus-mt-en-mr' 
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    #receive text from the user
    data = request.get_json()
    input_text = data['text']

    #making translation using ml model
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids, max_new_tokens=10000)
    final_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #returning the text
    return jsonify(final_text)

@app.route('/translate_mr_en', methods=['POST'])
def translate_mr_en():
    print('request received')
    #Loading ml model for Malyalam to English translation
    model_name = 'Helsinki-NLP/opus-mt-mr-en' 
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    #receive text from the user
    data = request.get_json()
    input_text = data['text']

    #making translation using ml model
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids, max_new_tokens=10000)
    final_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #returning the text
    return jsonify(final_text)

@app.route('/translate_ur_en', methods=['POST'])
def translate_ur_en():
    print('request received')
    #Loading ml model for Malyalam to English translation
    model_name = 'Helsinki-NLP/opus-mt-ur-en' 
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    #receive text from the user
    data = request.get_json()
    input_text = data['text']

    #making translation using ml model
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids, max_new_tokens=10000)
    final_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #returning the text
    return jsonify(final_text)

@app.route('/translate_en_ur', methods=['POST'])
def translate_en_ur():
    print('request received')
    #Loading ml model for Malyalam to English translation
    model_name = 'Helsinki-NLP/opus-mt-en-ur' 
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    #receive text from the user
    data = request.get_json()
    input_text = data['text']

    #making translation using ml model
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids, max_new_tokens=10000)
    final_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #returning the text
    return jsonify(final_text)



@app.route('/text_audio', methods=['POST'])
def text_audio():
        print("request received")
        data = request.get_json()
        text = data['text']
        #text = json.loads(data)['text']

        # Initialize the text-to-speech engine
        engine = pyttsx3.init()
        # Set the properties of the voice
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)  # change the index to select a different voice

        output_path = "output.wav"
       
        # Define a function to generate speech from input text and save it as an audio file
        def synthesize(text, output_path):
            # Generate speech from input text
            engine.save_to_file(text, output_path)
            engine.runAndWait()

        synthesize(text,output_path)

        # Set the headers for the response
        headers = {
            'Content-Disposition': f'attachment; filename=output.wav',
            'Content-Type': 'audio/wav',
        }
        print("sent")
        return make_response(send_file(output_path, as_attachment=True), headers)
        #return send_file('E:/git-clone/api/output.wav', as_attachment=True, attachment_filename='output.wav')

        #return jsonify(output_path)
     

#MongoDB storage system
@app.route('/store_data', methods = ['POST'])  
def store_data():
    username = request.json['username']
    id = request.json['id']
    title = request.json['title']
    content = request.json['content']
    time = request.json['time']
    user_data = {
            'username': username,
            'id' : id,
            'title': title,
            'content': content,
            'time': time,
        }
    result = user_collection_data.insert_one(user_data)
    inserted_id = result.inserted_id
    stored_user_data = user_collection_data.find_one({'_id': inserted_id})
    print(stored_user_data)
    print('data stored')
    return jsonify('success')

@app.route('/retrieve_data_username', methods = ['POST'])  
def retrieve_data_username():
    username = request.json['username']
    print(username)
    docs = user_collection_data.find({'username': username})
    doc_list = []
    for doc in docs:
        print(doc)
        doc_dict = {
            'id': doc['id'],
            'title': doc['title'],
            'content': doc['content'],
            'time': doc['time']
        }
        doc_list.append(doc_dict)
    print(type(doc_list))
    print(doc_list)
    return jsonify(doc_list)
 

@app.route('/retrieve_data_id', methods = ['POST'])  
def retrieve_data_id():
    id = request.json['id']
    docs = user_collection_data.find({'id': id})
    doc_list = []
    for doc in docs:
        print(doc)
        doc_dict = {
            'id': doc['id'],
            'title': doc['title'],
            'content': doc['content'],
            'time': doc['time']
        }
        doc_list.append(doc_dict)
        #print(file_list)
    return jsonify(doc_list)

@app.route('/delete_data_username', methods = ['POST'])  
def delete_data_username():
    username = request.json['username']
    id = request.json['id']
    title = request.json['title']
    result = user_collection_data.find_one_and_delete({'username': username,'id':id, 'title': title})
    print(result)
    if result:
        return jsonify('success')
    else:
        return jsonify('Fail, PLease retry!')


if __name__ == "__main__":
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
    
    print('MODEL LOADED')
    app.run(host='0.0.0.0', port=12345, debug = True)
    #app.run(host='127.0.0.1', port=0, debug=True)

#app.run()