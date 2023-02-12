import numpy as np
import argparse


def generate_keys(file_name_Alice, file_name_Bob, key_size, QBER):
    # key size in bytes
    # Generate random key for Alice out key_size bytes
    alice_key = np.random.randint(low=0, high=255, size=key_size, dtype=np.ubyte)
    # Generate Bob's key to be idenitcal as Alice's one
    bob_key = np.copy(alice_key)
    # Randomly generated bit flip errors for Bob's keys
    errors = np.random.choice(key_size, size=int(key_size*8*QBER),replace=False)
    print(f'number of errors: {np.size(errors)}')
    error_byte = errors    
    error_bits = np.random.randint(8, size=int(key_size*8*QBER))
    # Introduce bit errors in Bob's key
    for ebyt, ebit in zip(error_byte, error_bits):
        bob_key[ebyt] = change_bit(bob_key[ebyt], ebit)
    return (alice_key, bob_key)


# Flip the bit with an index bit_index
def change_bit(value, bit_index):
    return value ^ 1 << bit_index


def main(args):
    
    (key_size, file_name_Alice, file_name_Bob, QBER) = (args.key_size, args.alice, args.bob, args.qber)
    try:
        if QBER > 1.0 or QBER < 0:
            raise ValueError('QBER is invalid')
            
        if  key_size <=0 or key_size > 10**6:
            raise ValueError('Key_rate is invalid')
            
        (alice_key, bob_key) = generate_keys(file_name_Alice, file_name_Bob, key_size, QBER)
        # Save Alice's and Bob's and keys
        np.save(file_name_Alice, alice_key, allow_pickle=True, fix_imports=True)
        np.save(file_name_Bob, bob_key, allow_pickle=True, fix_imports=True)
            
    except ValueError as ex:
            print(f"Invalid range, {ex}")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-key_size', type=int, default=1024, help='Key size in bytes \(less than Megabyte\)')
    parser.add_argument('-qber', type=float, default=0.02, help='QBER', required=False)
    parser.add_argument('-alice', type=str, default='Alice-key', help='Alice\'s key name', required=False )
    parser.add_argument('-bob', type=str, default='Bob-key', help='Bob\'s key name', required=False)
    args = parser.parse_args()
    main(args)