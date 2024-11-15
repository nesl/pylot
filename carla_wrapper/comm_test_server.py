import socket
import pickle
import time
from utils.service import send_msg, recv_msg

def local_server():
    host = "0.0.0.0"
    port = 10000
    local_socket = socket.socket()
    local_socket.bind((host, port))
    
    print("Server is running...")
    local_socket.listen(10)
    local_conn, address = local_socket.accept()  

    while True:
        input_data = recv_msg(local_conn)
        
        try:
            server_timestamp = time.time()
            # Deserialize received data
            #input_message = pickle.loads(input_data)
            #print(input_data)
            #timestamp_send = input_message['timestamp_send']
            #rgb_frame = input_message['rgb_frame']
            #depth_frame = input_message['depth_frame']
            #pose = input_message['pose']  # Original sensor data is preserved

            # Simulate processing delay
            #time.sleep(0.05)  # 50ms processing delay
            
            server_timestamp = time.time()  # Timestamp after processing
        except (pickle.UnpicklingError, KeyError) as e:
            print("Error decoding input data:", e)
            continue

        # Send back server processing timestamp
        response = {'server_timestamp': time.time()-server_timestamp}
        serialized_response = pickle.dumps(response)
        send_msg(local_conn, serialized_response)

if __name__ == '__main__':
    local_server()
