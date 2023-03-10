import socket
import time
import numpy as np
import argparse
import ipaddress

HEADER = 64
PORT = 5050
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
QBER_ESTIMATE_MESSAGE = 'QBER'
ERROR_CORRECTION_MESSAGE = 'CORRECT'

class Bob:
    def __init__(self, key, ip, port):
        # time contains indices of the events.
        self.client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.client.connect((ip, port))
        self.key = key
        self.key_size = np.size(key)
        self.qber = 0
                
    def send_msg(self,msg):
        message = msg.encode(FORMAT)
        msg_length = len(message)
        send_length = str(msg_length).encode(FORMAT)
        send_length += b' '*( HEADER -len(send_length))
        self.client.send(send_length)
        self.client.send(message)
        print(self.client.recv(HEADER))
        
        
    def get_part_for_QBER_test(self,test_size):
        # Sample random part of key's bytes by their indeces
        test_key_indeces=np.random.choice(np.arange(0,np.size(self.key),dtype=np.uint32),test_size,replace=False)
        test_key = self.key[test_key_indeces]
        # Delete sampled keys for QBER test
        # self.key=np.delete(self.key,test_key_indeces)
        self.key_size = np.size(self.key)
        return (test_key,test_key_indeces)
    
    def send_QBER_test(self,sample_size):
        # Send some portion of the key for a QBER test
        test_size = int(np.size(self.key) * sample_size) 
        (test_key,test_key_indeces) = self.get_part_for_QBER_test(test_size)
        test_key_bytes = test_key.tobytes(order='C')
        test_key_indeces_bytes = test_key_indeces.tobytes(order='C')
        # Initiate the QBER estimation
        self.send_msg(QBER_ESTIMATE_MESSAGE)
        # Size of sent data
        indeces_array_size = int(len(test_key_indeces_bytes))
        data_array_size=int(len(test_key_bytes))
        
        
        self.client.send(data_array_size.to_bytes(4, 'big'))
        self.client.send(indeces_array_size.to_bytes(4, 'big'))
        
        # send key test 
        self.client.send(test_key_bytes)
        self.client.send(test_key_indeces_bytes)
        self.qber = np.frombuffer(self.client.recv(16))[0]
        print(f"qber={self.qber}")
        
    
    
    
    def number_of_ones(self,byte):
    
        arr=[np.ubyte(0x80), np.ubyte(0x40), np.ubyte(0x20), np.ubyte(0x10),
         np.ubyte(0x08), np.ubyte(0x04), np.ubyte(0x02), np.ubyte(0x01)]
        
        return np.count_nonzero(np.bitwise_and(byte,arr))    
    
    
    
    def byte_array_parity(self,array):
        # Calculate a parity of array of bytes
        # 0 for even partity
        # 1 for odd
        if np.shape(array) == ():
            s = self.number_of_ones(array)
        else:    
            s=0
            for x in array:
                s+=self.number_of_ones(x)
        return s % 2    
            
            
        
    def cascade(self, it_number):
        
        self.send_msg(ERROR_CORRECTION_MESSAGE)
        self.client.send(it_number.to_bytes(32, 'big'))
        
        for it in np.arange(0,it_number):
            
            # Split the key in chunks of such size  
            # that there is only one error in a chunk
            # on average
            self.qber = np.around(self.qber,decimals=5)
            trun_size = int((2**it)*0.73/self.qber/8)
            if trun_size >= self.key_size:
                trun_size = int(trun_size /2)
            print(f'trun_size: {trun_size}')    
            self.client.send(trun_size.to_bytes(32, 'big'))
            # generate a random array of key bytes indeces
            shuffle_pos=np.arange(0,self.key_size,dtype=np.uint32)
            np.random.shuffle(shuffle_pos)
            # divide the indeces array into the chunks of size trun_size
            chunks = np.array(np.array_split(shuffle_pos, trun_size))
            chunk_parities = np.zeros(np.size(chunks),dtype=np.ubyte)
            # calculate a parity for the chunks 
            for i, chunk in np.ndenumerate(chunks):
                chunk_parities[i[0]] = self.byte_array_parity(self.key[chunk])
                chunk_bytes = chunk.tobytes(order='C')
                p=int(len(chunk_bytes))
                self.client.send(p.to_bytes(32, 'big'))
                self.client.send(chunk_bytes)
            chunk_parities_bytes = chunk_parities.tobytes(order='C')   
            parities_size = len(chunk_parities.tobytes(order='C'))
            self.alice_parities = np.frombuffer(self.client.recv(parities_size),dtype=np.ubyte)
            
            errors = np.equal(self.alice_parities, chunk_parities)
            errors = np.logical_not(errors)
            erroneous_chunks = np.array(np.argwhere(errors).flatten(),dtype=np.uint16)

            
            # Locate error and correct error by cascade algorithm
            self.locate_errors(np.copy(chunks), erroneous_chunks, it)
            np.save('corrected-Bob-key', self.key, allow_pickle=True, fix_imports=True)
            
    def locate_errors(self, chunks, erroneous_chunks, it):
        # send Alice number of erroneous chunks
        err_number = int(np.size(erroneous_chunks))
        self.client.send(err_number.to_bytes(16, 'big'))
        # send Alice erroneous chunks
        self.client.send(erroneous_chunks.tobytes(order='C'))
        
        error_chunks = []
        for err_chunk in erroneous_chunks:
            error_chunks.append(chunks[err_chunk])
        
        self.binary_search(error_chunks,it)

            
    def binary_search(self, error_chunks, it):
        # search for errors by dividing the key in half 
        # and comparing a parity of each halfs
        if len(error_chunks) == 0:
            print(f'No errors were found after {it}-th iteration')
            return 0
        
        
        elif (np.size(error_chunks[0])) == 1:
            # correct bit error 
            for err_chunk in error_chunks:
                bad_bit = self.find_bit_in_byte(err_chunk)
                self.key[err_chunk] =  (self.key[err_chunk] ^ bad_bit)
            
            
            print(f'{it}-th iteration is over! {len(error_chunks)} errors are corrected!')
            
        elif (np.size(error_chunks[0])) == 0:
            print(f'No errors were found')
        
        else:
            new_chunks = []
            for err_chunk in error_chunks:
                new_chunk = self.receive_alice_parities(err_chunk)
                new_chunks.append(new_chunk)

            self.binary_search(new_chunks,it)
        
        
    def find_bit_in_byte(self, chunk):
        # find false bit in a byte of the key defined by chunk.
        if np.shape(chunk) == ():
                byte = self.key[chunk]
        else:
                byte = self.key[chunk[0]]
        left_4_bits = byte & np.ubyte(0xf0)
        right_4_bits = byte & np.ubyte(0x0f)
        
        if self.left_partity(left_4_bits, right_4_bits):
            # send 0 to Alice to indicate that left part is used
            #      1 to Alice to indicate that right part is used
            error_part = int(0)
            self.client.send(error_part.to_bytes(4, 'big'))
            left_2_bits = byte & np.ubyte(0xc0)
            right_2_bits = byte & np.ubyte(0x30)
            
            if self.left_partity(left_2_bits, right_2_bits):
                error_part = int(0)
                self.client.send(error_part.to_bytes(4, 'big'))
                left_1_bits = byte & np.ubyte(0x80)
                right_1_bits = byte & np.ubyte(0x40)
                
                if self.left_partity(left_1_bits, right_1_bits):
                    return np.ubyte(0x80)
                else:
                    return np.ubyte(0x40)
                
                
            else:
                error_part = int(1)
                self.client.send(error_part.to_bytes(4, 'big'))
                left_1_bits = byte & np.ubyte(0x20)
                right_1_bits = byte & np.ubyte(0x10)
            
                if self.left_partity(left_1_bits, right_1_bits):
                    return np.ubyte(0x20)
                else:
                    return np.ubyte(0x10)
                            
        else:
            
            error_part = int(1)
            self.client.send(error_part.to_bytes(4, 'big'))
            left_2_bits = byte & np.ubyte(0x0c)
            right_2_bits = byte & np.ubyte(0x03)
            
            if self.left_partity(left_2_bits, right_2_bits):
                error_part = int(0)
                self.client.send(error_part.to_bytes(4, 'big'))
                left_1_bits = byte & np.ubyte(0x08)
                right_1_bits = byte & np.ubyte(0x04)
                
                if self.left_partity(left_1_bits, right_1_bits):
                    return np.ubyte(0x08)
                else:
                    return np.ubyte(0x04)       
            else:
                error_part = int(1)
                self.client.send(error_part.to_bytes(4, 'big'))
                left_1_bits = byte & np.ubyte(0x02)
                right_1_bits = byte & np.ubyte(0x01)
                
                if self.left_partity(left_1_bits, right_1_bits):
                    return np.ubyte(0x02)
                else:
                    return np.ubyte(0x01)       
            
            
        
    
    
    
    def left_partity(self, left_bits, right_bits):
        # returns True if error in left parts
        # returns False if error in right parts
        bob_left_parity =  int(self.number_of_ones(left_bits) % 2)
        bob_right_parity = int(self.number_of_ones(right_bits) % 2)
        alice_left_parity =  int.from_bytes(self.client.recv(4),'big')
        alice_right_parity =  int.from_bytes(self.client.recv(4),'big')
        if alice_left_parity != bob_left_parity:
            return True
        elif alice_right_parity != bob_right_parity:
            return False
    
    
    def receive_alice_parities(self, error_chunks):
        
        x = np.array(np.array_split(error_chunks, 2))
        bob_left_parity = int(self.byte_array_parity(self.key[x[0]]))
        bob_right_parity = int(self.byte_array_parity(self.key[x[1]]))
        alice_left_parity =  int.from_bytes(self.client.recv(4),'big')
        alice_right_parity =  int.from_bytes(self.client.recv(4),'big')
        
        if alice_left_parity != bob_left_parity:
            error_part = int(0)
            self.client.send(error_part.to_bytes(4, 'big'))
            return x[0]
        elif alice_right_parity != bob_right_parity:
            error_part = int(1)
            self.client.send(error_part.to_bytes(4, 'big'))
            return x[1]
        else:
            return None # chunks[err_chunk]
        
        
        
def str_2_bool(v):
    return v.lower() in ("yes", "true", "t", "1")        
    
    
    
    
        
def start_communication(key, sample_size, it_number, ip, port, qber_test):
    
    if qber_test:
        bob=Bob(key,ip,port)
        bob.send_QBER_test(sample_size)
        bob.send_msg(DISCONNECT_MESSAGE)
        
    else:
        bob=Bob(key,ip,port)
        bob.send_QBER_test(sample_size)
        if bob.qber == 0:
            print('QBER is zero, no reconciliation is needed')
            bob.send_msg(DISCONNECT_MESSAGE)
        else:    
            bob.cascade(it_number)
            bob.send_msg(DISCONNECT_MESSAGE)
                
        

def main(args):
    
    try:
        if args.it <= 0:
            raise ValueError('Number of iterations should larger than 0.') 
        
        if args.sample_size <=0 or args.sample_size >=1:
            raise ValueError('sample size should be between 0 and 1.')
            
        ip_address = args.ip 
        ip = ipaddress.ip_address(ip_address)    
        port = args.port
        qber_test = str_2_bool(args.qber)
        
        if port > 65535 or port < 0 :
            raise ValueError('invalid port number')
        
        key=np.load(file=args.key)
        start_communication(key, args.sample_size, args.it, ip_address, port, qber_test)
        
    except FileNotFoundError:
        print("Key file is not found.")
    except ValueError as e:
        print(str(e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-key', type=str, default='Bob-key.npy', help='Name of a key file')
    parser.add_argument('-qber', type=str, default='False', help='Perform only qber test, default is False')
    parser.add_argument('-ip', type=str, default='127.0.1.1', help='IP address of the Alice\'s server')
    parser.add_argument('-port', type=int, default=5050, help='port number of the Alice\'s server')
    parser.add_argument('-it', type=int, default=10, help='Number of iterations in cascade algorithm')
    parser.add_argument('-sample_size', type=float, default=0.1, help='Portion of key size for QBER estimation, defailt is 10% or 0.1')
    
    args = parser.parse_args()
    main(args)
