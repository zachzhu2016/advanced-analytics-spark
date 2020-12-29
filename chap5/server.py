import socket, time, sys, json

HOST = ''
PORT = 9999

with open('AAS_CH5/kddcup.testdata', newline='') as f:
    lines = f.readlines()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.listen(2)
        conn, addr = s.accept()
        with conn:
            while True:
                for line in lines: 
                    conn.sendall(line.encode())
                    time.sleep(0.1)

