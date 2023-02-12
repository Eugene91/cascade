import socket
import threading
import numpy as np
import argparse
import ipaddress

#  size in bytes of generic socket message
HEADER = 64

FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
QBER_ESTIMATE_MESSAGE = 'QBER'
ERROR_CORRECTION_MESSAGE = 'CORRECT'


    

class Alice:
    
    def __init__(self,key,ip,port):
        # self.ip = socket.gethostbyname(socket.gethostname())
        self.server= socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.ip = ip
        self.port = port
        self.ADDR=(ip,port)
        self.alice_key = key
        
    def start_server(self):
        self.server.bind((self.ip,self.port))
        self.server.listen()
        print(f"[LISTENING] Server is listening on {self.ip}:{self.port}")
        while True:
            conn, addr = self.server.accept()
            thread = threading.Thread(target=self.handle_client,args=(conn,addr))
            thread.start()
            print(f"[ACTIVE CONNECTION] {threading.activeCount()-1}")
        
    def prepare_str(self,msg):
        message = msg.encode(FORMAT)
        if len(message) < HEADER:
            message += b''*( HEADER -len(msg))
        return message
    
    
    
    def qber_comm(self,conn):
        # receive two bytes to determine size of the message
        conn.send("Ready to receive size of a test buffer".encode(FORMAT))
        size_qber_sample = int.from_bytes(conn.recv(4),'big')
        size_qber_sample_indeces = int.from_bytes(conn.recv(4),'big')
        message = self.prepare_str(f"ready to receive data with {size_qber_sample} bytes and data's indeces of {size_qber_sample_indeces} bytes")
        #conn.send(message)
        
        sample = np.frombuffer(conn.recv(size_qber_sample), dtype=np.ubyte)
        sample_indeces = np.frombuffer(conn.recv(size_qber_sample_indeces), dtype=np.uint32)
                
        self.qber = self.calculate_qber(bob_key=sample,indeces=sample_indeces)
        print(f"qber:{self.qber}")
        conn.send(np.ndarray.tobytes(np.array([self.qber])))
        self.new_alice_key= self.alice_key #np.delete(self.alice_key,sample_indeces)
        self.new_key_size = np.size(self.new_alice_key)
        
        
        
    def calculate_qber(self,bob_key,indeces):
        errors = np.bitwise_xor(bob_key,self.alice_key[indeces])
        s=0
        for error in errors[np.flatnonzero(errors)]:
            s+=self.number_of_ones(error)
            
        return s/(8*np.size(bob_key))
    
    
    
    def number_of_ones(self,byte):
    
        arr=[np.ubyte(0x80), np.ubyte(0x40), np.ubyte(0x20), np.ubyte(0x10),
         np.ubyte(0x08), np.ubyte(0x04), np.ubyte(0x02), np.ubyte(0x01)]
        
        return np.count_nonzero(np.bitwise_and(byte,arr))
    
    
    def error_correction(self,conn):
        
        #correction = True
        # Get number of iterations
        it_number = int.from_bytes(conn.recv(32),'big')
        for it in np.arange(0,it_number):
            trun_size = int.from_bytes(conn.recv(32),'big')
            if trun_size == 0:
                print(f'error: {trun_size}')
            print(f'trun_size: {trun_size}')
            bob_chunks=np.zeros(self.new_key_size,dtype=np.uint32)
            bob_chunks=np.array_split(bob_chunks, trun_size)
            alice_chunk_parities = np.zeros(np.size(bob_chunks),dtype=np.ubyte)
            x=0
            for i, chunk in np.ndenumerate(bob_chunks):
                chunk_size = int.from_bytes(conn.recv(32),'big')
                received_chunk = np.frombuffer(conn.recv(chunk_size),dtype=np.uint32)
                bob_chunks[i[0]] = received_chunk
                alice_chunk_parities[i[0]]=self.byte_array_parity(self.new_alice_key[received_chunk])
                
            alice_chunk_parities = alice_chunk_parities.tobytes(order='C')
            conn.send(alice_chunk_parities)
            
            num_err = 0
            num_err = int.from_bytes(conn.recv(16),'big')
            erroneous_chunks=np.frombuffer(conn.recv(num_err*2),dtype=np.uint16)
            
            error_chunks= []
            for err_chunk in erroneous_chunks:
                error_chunks.append(bob_chunks[err_chunk])
                
            byte_error_correction = True
            
            while byte_error_correction:
                    
                if len(error_chunks) == 0:
                    byte_error_correction = False
                    print(f'No errors were found after {it}-th iteration')
                    break
                
                new_chunks = []
                for err_chunk in error_chunks:
                    new_chunk =  self.send_alice_parities(conn, err_chunk)
                    new_chunks.append(new_chunk)
                
                
                if np.size(new_chunks[0]) == 1:
                    error_chunks = new_chunks
                    byte_error_correction = False
                else:
                    error_chunks = new_chunks
                    
            
            for err_chunk in error_chunks:
                if np.size(err_chunk) == 1:
                    self.find_bit_in_byte(conn, err_chunk)
                elif np.size(err_chunk)>1:
                    self.find_bit_in_byte(conn, err_chunk[0])
                else:
                    pass
                    
        #np.save('corrected-Alice-key', self.new_alice_key, allow_pickle=True, fix_imports=True)         
                    
                
    def find_bit_in_byte(self, conn, chunk):
        # find false bit in byte
        byte = self.alice_key[chunk]
        left_4_bits = byte & np.ubyte(0xf0)
        right_4_bits = byte & np.ubyte(0x0f)
        self.exchange_partity(conn, left_4_bits, right_4_bits)
        right_part = int.from_bytes(conn.recv(4),'big')
        # True if right part
        # False if left part
        if right_part:
            left_2_bits = byte & np.ubyte(0x0c)
            right_2_bits = byte & np.ubyte(0x03)
            self.exchange_partity(conn, left_2_bits, right_2_bits)
            right_part = int.from_bytes(conn.recv(4),'big')
            if right_part:
                left_1_bits = byte & np.ubyte(0x02)
                right_1_bits = byte & np.ubyte(0x01)
                self.exchange_partity(conn, left_1_bits, right_1_bits)
                
            else:
                left_1_bits = byte & np.ubyte(0x08)
                right_1_bits = byte & np.ubyte(0x04)
                self.exchange_partity(conn, left_1_bits, right_1_bits)
            
            
        else:
            left_2_bits = byte & np.ubyte(0xc0)
            right_2_bits = byte & np.ubyte(0x30)
            self.exchange_partity(conn, left_2_bits, right_2_bits)
            right_part = int.from_bytes(conn.recv(4),'big')
            
            if right_part:
                left_1_bits = byte & np.ubyte(0x20)
                right_1_bits = byte & np.ubyte(0x10)
                self.exchange_partity(conn, left_1_bits, right_1_bits)
                
            else:
                left_1_bits = byte & np.ubyte(0x80)
                right_1_bits = byte & np.ubyte(0x40)
                self.exchange_partity(conn, left_1_bits, right_1_bits)
                
                
        
        
    def exchange_partity(self, conn, left_bits, right_bits):
        # returns True if error in left parts
        # returns False if error in left parts
        alice_left_parity =  int(self.number_of_ones(left_bits) % 2)
        alice_right_parity = int(self.number_of_ones(right_bits) % 2)
        conn.send(alice_left_parity.to_bytes(4, 'big'))
        conn.send(alice_right_parity.to_bytes(4, 'big'))
    
    
    
    
    def send_alice_parities(self, conn, chunks):

            
            x = np.array(np.array_split(chunks, 2))
            alice_left_parity = self.byte_array_parity(self.alice_key[x[0]])
            alice_right_parity = self.byte_array_parity(self.alice_key[x[1]])
            conn.send(alice_left_parity.to_bytes(4, 'big'))
            conn.send(alice_right_parity.to_bytes(4, 'big'))
            part = int.from_bytes(conn.recv(4),'big') 
            if part == 0:
                return x[0]
            else:
                return x[1]
            
    def byte_array_parity(self,array):
        # Calculate a parity of array of bytes
        # 0 for even partity
        # 1 for odd
        s=0
        for x in array:
            s+=self.number_of_ones(x)
        return s % 2   
        
    
    
    def handle_client(self, conn, addr):
        print(f"[New CONNECTION] {addr} connected")
        connected = True
        while connected:
            header_msg = conn.recv(HEADER).decode(FORMAT)
            msg_length = len(header_msg)
            header_msg = conn.recv(msg_length).decode(FORMAT)
            
            if header_msg == DISCONNECT_MESSAGE:
                conn.send("Bye!".encode(FORMAT))
                connected = False

            elif header_msg == QBER_ESTIMATE_MESSAGE:
                self.qber_comm(conn)

            elif header_msg == ERROR_CORRECTION_MESSAGE:
                message = self.prepare_str("correcting errors!")
                conn.send(message)
                self.error_correction(conn)
                
            else:    
                message = self.prepare_str("Available commands: 1.!DISCONNECT 2.QBER 3.CORRECT")
                conn.send(message)

            print(f'header message= {header_msg}')    

        conn.close()
        


        
        
        
def main(args):
    try:
        key=np.load(file=args.key_name)
        key_size = np.size(key)
        
        ip_address = args.ip 
        ip = ipaddress.ip_address(ip_address)
        port = args.port 
        if port > 65535 or port < 0 :
            raise ValueError('invalid port number')
        
        alice = Alice(key=key,ip=ip_address,port=port)
        alice.start_server()
        
    except FileNotFoundError:
        print("Key file not found")
    except ValueError as e:
        print(str(e))
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-key_name', type=str, default='Alice-key.npy', help='Name of a key file')
    parser.add_argument('-ip', type=str, default='127.0.1.1', help='An ip address of the server to run on')
    parser.add_argument('-port', type=int, default=5050, help='A network port to use')
    
    args = parser.parse_args()
    main(args)