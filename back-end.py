from transformers import AutoTokenizer, AutoModel #this
import boto3 #this
import os
import zipfile
import re
import torch # this
import numpy as np # this
import faiss # this
import tempfile
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
import queue
import time
import threading
import requests
import re

MAX_THREADS = 5
MESSAGE_QUEUE = queue.Queue(maxsize=MAX_THREADS)
# queue_url = os.environ['QUEUE_URL']
queue_url = "https://sqs.us-east-2.amazonaws.com/257394469414/MessageQueue.fifo"

# input_bucket = os.environ['INPUT_BUCKET']
input_bucket = "5411-copilot-project-v2"
# vector_bucket = os.environ['VECTOR_BUCKET']
vector_bucket = "5411-vector-db-store"
# socket_endpoint = os.environ['SOCKET_ENDPOINT']
socket_endpoint = "https://8e8h4nm355.execute-api.us-east-2.amazonaws.com/production" 
# connections_table = os.environ['CONNECTIONS_TABLE']
connections_table = "UserSocketConnections"
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
s3 = boto3.client('s3', region_name='us-east-2')
sagemaker = boto3.client('sagemaker-runtime', region_name='us-east-2')
sqs = boto3.client('sqs', region_name='us-east-2')
client = boto3.client('apigatewaymanagementapi', endpoint_url=socket_endpoint, region_name='us-east-2')
dynamodb = boto3.client('dynamodb', region_name='us-east-2')
base_path = "."

def fetch_code_from_s3(file_path):

    parts = file_path.replace("s3://", "").split("/")

    bucket_name = parts[0]
    userId = parts[1]
    object_key = parts[2]
    local_path = os.path.join(base_path, userId)
    os.makedirs(local_path, exist_ok=True)

    print("===================================")
    print(f"Downloading: {object_key} from bucket {bucket_name} to local path {local_path}")
    
    s3.download_file(bucket_name, f"{userId}/{object_key}", os.path.join(local_path, object_key))
    
    return local_path, object_key, userId

def process_path(local_path, object_key):

    print("===================================")
    print("Local Path:", local_path)
    extract_path = extract_zip(local_path, object_key) if object_key.endswith('.zip') else local_path
    index, metadata = process_repository(extract_path)
    print("Extracted File Path:", extract_path)

    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    
    return index, metadata
        
def extract_zip(local_path, object_key):

    extract_path = os.path.join(local_path, 'extracted')
    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(os.path.join(local_path, object_key), 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    return extract_path

def process_repository(extract_path):

    embeddings = []
    metadata = []

    for root, dirs, files in os.walk(extract_path):
        for file in files:
            
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, extract_path)
            chunks = process_code_file(file_path, rel_path)

            for chunk in chunks:
                embedding = get_embedding(chunk)

                # Ensure embedding is normalized for cosine similarity
                embedding = embedding / np.linalg.norm(embedding)
                
                embeddings.append(embedding)
                metadata.append(chunk)
    
    dimensions = embeddings[0].shape[0]
    index = faiss.IndexFlatIP(dimensions)

    embeddings_array = np.array(embeddings).astype('float32')
    index.add(embeddings_array)

    return index, metadata

def process_code_file(file_path, rel_path):

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        code = f.read()
    
    chunks = [code]

    # pattern = re.compile(pattern = re.compile(r'^\s*(def|class|function|public|private|protected|static|\w+\s*=\s*(async\s*)?(function|\(.*\)\s*=>))', re.IGNORECASE))

    # current_chunk = []
    # current_chunk_start = 0
    # in_chunk = False
    # chunk_name = "unnamed"
    # chunk_type = "code"

    # lines = code.splitlines()
    # print(f"Processing file: {file_path}, total lines: {len(lines)}")

    # for i, line in enumerate(lines):
    #     if pattern.match(line):
    #         # start of a new chunk
    #         if in_chunk:
    #             chunk_code = "\n".join(current_chunk)
    #             chunks.append({
    #                 "code": chunk_code,
    #                 "display_code": chunk_code,
    #                 "file_path": rel_path,
    #                 "start_line": current_chunk_start + 1,
    #                 "end_line": i,
    #                 "type": chunk_type,
    #                 "name": chunk_name
    #             })
                
    #         current_chunk = [line]
    #         current_chunk_start = i
    #         in_chunk = True
    #         chunk_name = extract_chunk_name(line.lower())
    #         chunk_type = extract_chunk_type(line.lower())
    #     elif in_chunk:
    #         # continue the current chunk
    #         current_chunk.append(line)
    
    # if in_chunk:
    #     chunk_code = "\n".join(current_chunk)
    #     chunks.append({
    #         "code": chunk_code,
    #         "display_code": chunk_code,
    #         "file_path": rel_path,
    #         "start_line": current_chunk_start + 1,
    #         "end_line": len(lines),
    #         "type": chunk_type,
    #         "name": chunk_name
    #     })
    
    # if not chunks:
    #     chunks = chunk_by_size(lines, rel_path)
    
    return chunks

# def chunk_by_size(lines, file_path, chunk_size=50):
#     chunks = []
#     for i in range(0, len(lines), chunk_size):
#         chunk_lines = lines[i:i+chunk_size]
#         chunks.append({
#             "code": "\n".join(chunk_lines),
#             "display_code": "\n".join(chunk_lines),
#             "file_path": file_path,
#             "start_line": i + 1,
#             "end_line": min(i + chunk_size, len(lines)),
#             "type": "chunk",
#             "name": f"chunk_{i//chunk_size + 1}"
#         })
#     return chunks

# def extract_chunk_name(line):

#     match = re.search(r'(def|class|function)\s+(\w+)', line)
#     if match:
#         return match.group(2)
#     return "unnamed"

# def extract_chunk_type(line):

#     if "class" in line:
#         return "class"
#     if "def" in line or "function" in line:
#         return "function"
#     return "code"

def get_embedding(code):

    inputs = tokenizer(
        code,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
                             
     # Use the [CLS] token embedding as the text representation
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings[0]

def save_vectors_to_s3(index, metadata, userId, bucket_name):

    print("===================================")
    print(f"Saving vectors to S3 bucket: {bucket_name}, userId: {userId}")

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as index_file:

        faiss.write_index(index, index_file.name)
        index_file.flush()

        s3.upload_file(
            index_file.name,
            bucket_name,
            f"{userId}/index.faiss"
        )
    
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as metadata_file:

        json.dump(metadata, metadata_file)
        metadata_file.flush()
        metadata_file.seek(0)

        s3.upload_file(
            metadata_file.name,
            bucket_name,
            f"{userId}/metadata.json"
        )

def download_github_repo(repo_url, userId):
    
    print("===================================")
    print(f"Downloading GitHub repository: {repo_url} for userId: {userId}")

    local_path = os.path.join(base_path, userId)
    os.makedirs(local_path, exist_ok=True)

    os.system(f"git clone {repo_url} {local_path}")

    return local_path

def process_query(query, userId, k=3):

    index, metadata = load_vectors_from_s3(userId)

    query_embedding = get_embedding(query)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    distances, indices = index.search(np.array([query_embedding]), k)

    results = []

    for i, index in enumerate(indices[0]):
        if(index < len(metadata)):
            result = {
                "code": metadata[index],
            }
            results.append(result)

    context = build_llm_context(results)

    prompt, context = create_llm_prompt(query, context)

    print("Prompt for LLM:")
    print(prompt)

    llm_response = get_llm_response(prompt, context)

    return llm_response

def build_llm_context(results):

    print("===================================")
    print("Building context for LLM response...")
    context = "Here are the relevant code snippets:\n\n"

    for i, result in enumerate(results):
        context += f"Relevant Code:\n{result['code']}\n"
    
    return context.strip()


def load_vectors_from_s3(userId, bucket_name=vector_bucket):
    
    print("===================================")
    print(f"Loading vectors from S3 bucket: {bucket_name}, userId: {userId}")

    index_file = tempfile.NamedTemporaryFile(delete=False)
    metadata_file = tempfile.NamedTemporaryFile(delete=False)
    
    try:
        s3.download_file(
            bucket_name,
            f"{userId}/index.faiss",
            index_file.name
        )

        s3.download_file(
            bucket_name,
            f"{userId}/metadata.json",
            metadata_file.name
        )

        index = faiss.read_index(index_file.name)

        with open(metadata_file.name, 'r') as f:
            metadata = json.load(f)
            
        return index, metadata
    finally:
        os.unlink(index_file.name)
        os.unlink(metadata_file.name)


def create_llm_prompt(query, context):
    print("===================================")
    print("Creating LLM prompt...")

    prompt = f"""You are an expert code assistant who has knowledge of codes repositories.

    As an assistant your task is to: 
        1. Answer the USER QUESTION defined below directly without using any flowery language or jargon.
        2. Include any insights about code patterns or potential improvements when relevant
        3. Use context to answer the questions, and if the context is not relevant to the question, please say "Sorry, I am not able to understand your query at the moment" or "I cannot help with that".
        4. Keep the response concise and to the point, avoiding unnecessary elaboration or repetition.
        5. You MUST stop generation once the answer has been fully given. Do not repeat the answer or generate filler text.

        Context:
        {context}

        USER QUESTION: 
        {query}

        Answer:"""
    
    return prompt, context

def get_llm_response(prompt, context):

    print("===================================")
    print("Getting LLM response...")

    body = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.3,
            "top_p": 0.9
        }
    }

    response = sagemaker.invoke_endpoint(
        EndpointName = "jumpstart-dft-llama-codellama-7b-in-20250601-160654",
        ContentType = 'application/json',
        Body = json.dumps(body)
    )

    response_content = response['Body'].read().decode()
    print("LLM Response:", response_content)
    response_content = json.loads(response_content)
    return response_content['generated_text']

def remove_connection_from_db(userId):
    
    try:
        dynamodb.delete_item(
            TableName = connections_table,
            Key = {
                'userId': {'S': userId}
            }
        )
        print(f"Connection for user {userId} removed successfully.")

    except dynamodb.exceptions.ResourceNotFoundException:
        print(f"Connection for user {userId} not found in the database.")

def send_response_to_connection(response, userId, connectionId=None):
        
    if connectionId is None:
        
        try:
            dynamodb_response = dynamodb.get_item(
                TableName = connections_table,
                Key = {
                    'userId': {'S': userId}
                }
            )

            connectionId = dynamodb_response['Item']['connectionId']['S']

        except dynamodb.exceptions.ResourceNotFoundException:
            print(f"User ({userId}) is already disconnected.")
            return
        
    try:

        print("Response to send:", response)

        client.post_to_connection(
            ConnectionId=connectionId,
            Data=json.dumps(response).encode('utf-8')
        )

    except client.exceptions.GoneException:
            print(f"User ({userId}) is disconnected, removing it from the database.")
            remove_connection_from_db(userId)

def is_repo_under_size_limit(repo_url, max_size_mb=25):
    match = re.match(r'https://github.com/([^/]+)/([^/]+)', repo_url)
    if not match:
        return False  

    owner, repo = match.groups()
    api_url = f'https://api.github.com/repos/{owner}/{repo}'

    headers = {
        'Accept': 'application/vnd.github.v3+json'
    }

    response = requests.get(api_url, headers=headers)
    if response.status_code != 200:
        print(f"GitHub API error: {response.status_code}")
        return False  # treat error as rejection

    data = response.json()
    size_kb = data.get('size', 0)

    return size_kb <= max_size_mb * 1024

def handle_message(message):

    message_body = json.loads(message['Body'])
    message_type = message_body.get('type')

    if(message_type == 'FILE_PATH'):

        local_path, object_key, userId = fetch_code_from_s3(message_body.get('url'))

        assert userId == message_body.get('userId'), "userId is incorrect in the s3"

        index, metadata = process_path(local_path, object_key)
        save_vectors_to_s3(index, metadata, userId, vector_bucket)
        data = {
            "type": "PROCESSING_COMPLETE",
        }
        send_response_to_connection(data, userId)

    elif(message_type == 'URL'):
        
        userId = message_body.get('userId')
        repo_url = message_body.get('url')

        if not is_repo_under_size_limit(repo_url):
            data = {
                "type": "ERROR",
                "message": "Repository size exceeds the limit of 25MB."
            }
            send_response_to_connection(data, userId)
        else:

            local_path = download_github_repo(repo_url, userId)
            index, metadata = process_repository(local_path)

            if os.path.exists(local_path):
                shutil.rmtree(local_path)

            save_vectors_to_s3(index, metadata, userId, vector_bucket)
            data = {
                "type": "PROCESSING_COMPLETE",
            }
            send_response_to_connection(data, userId)

    else:
        # type == 'QUERY'
        query = message_body.get('query')
        userId = message_body.get('userId')
        connectionId = message_body.get('connectionId')

        response = process_query(query, userId)
        data = {
            "type": "RESULT",
            "data": response
        }
        send_response_to_connection(data, userId, connectionId)

        # except Exception as e:
        #     print(f"Error processing SQS message: {e}")

def poll_sqs():

    queue_attributes = sqs.get_queue_attributes(
        QueueUrl = queue_url,
        AttributeNames = ['All']
    )
    print("Queue Attributes:", queue_attributes)

    wait_time_seconds = int(queue_attributes['Attributes'].get('ReceiveMessageWaitTimeSeconds', 5))
    visibility_timeout = int(queue_attributes['Attributes'].get('VisibilityTimeout', 180))

    while True:
        if MESSAGE_QUEUE.full():
            time.sleep(2)
            continue

        response = sqs.receive_message(
            QueueUrl = queue_url,
            MaxNumberOfMessages = MAX_THREADS,
            WaitTimeSeconds = wait_time_seconds,
            VisibilityTimeout = visibility_timeout
        )

        messages = response.get('Messages', [])
        
        for message in messages:
            MESSAGE_QUEUE.put(message)

def worker_loop():

    while True:
        message = MESSAGE_QUEUE.get()
        if message is None:
            continue

        try:
            handle_message(message)
        except Exception as e:
            print(f"Error processing message: {e}")
        finally:
            # Delete the message from the queue after processing
            sqs.delete_message(
                QueueUrl = queue_url,
                ReceiptHandle = message['ReceiptHandle']
            )
            print(f"Message {message['MessageId']} processed and deleted from the queue.")
            MESSAGE_QUEUE.task_done()


def main():

    threading.Thread(target=poll_sqs, daemon=True).start()

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        for _ in range(MAX_THREADS):
            executor.submit(worker_loop)

main()