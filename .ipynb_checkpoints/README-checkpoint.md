# Cascade error correction
Simple implementation of [cascade reconciliation algorithm](https://www.example.com)  with python numpy and TCP/IP sockets.

# Details
- Before reconciliation Alice and Bob share a sifted key being a correlated fixed size byte array. 
- For test purpose key-pair with given QBER can be generated with a generate_key_pair.py script.
- Script Alice-server.py runs a server with given IP address and port that represent an Alice with sifted key.
- Script Bob-client.py runs a Bob's client application that initiates reconciliation with the server.
- After reconciliation the key file named corrected-Bob-key.npy will be created.
----------
# Example 
- Copy repository
`git clone https://github.com/Eugene91/cascade`
- Create a test key pair with 3% of QBER and a maximum of a single error per byte
`python generate_key_pair.py -QBER 0.03 `
- Start Alice's server:
`python alice_server.py`
- Start reconciliation procedure with 10 iterations by 
`python bob-client.py -it 10`
- Check QBER after the reconciliation:
`python bob-client.py -qber True -key corrected-Bob-key.npy`
----------
The probabalistic nature of the cascade protocol may not provide full error correction in the first run,
hence several runs may be required to achieve 0 bit errors. 


# TODO:
- Introduce the verification of equal key size between Alice and Bob
- Remove a partion of the key being used for qber test
- Unpack bits from bytes and shuffle them to ease a contrain of a single error in a byte