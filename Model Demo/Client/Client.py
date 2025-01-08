import socket
import os
import sys

def choose_audio_file(demo_path):
    """
    Allows the user to choose an audio file by either:
    1. Manually selecting the file using a file dialog, or
    2. Entering the file path directly in the terminal.
    """
    print("1. Select audio file manually")
    print("2. Enter audio file path")
    print("3. Default file path")
    choice = input("Choose an option (1/2/3):")

    if choice == "1":
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename

        Tk().withdraw()  # Prevent the Tkinter GUI window from appearing
        file_path = askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.ogg")])
        if not file_path:
            print("No file selected.")
            exit(1)
        return file_path
    elif choice == "2":
        # Allow the user to manually input the file path
        file_path = input("Enter the full path of the audio file: ").strip()
        if not os.path.exists(file_path):
            print("Invalid file path.")
            exit(1)
        return file_path
    elif choice == "3":
        file_path = demo_path
        return file_path
    else:
        # Handle invalid input
        print("Invalid choice.")
        exit(1)




def send_audio_file(file_path, host='127.0.0.1', port=55451):
    """
    Connects to the server and sends the selected audio file for processing.
    Receives the result from the server and displays it to the user.
    """
    # Manually flushing sys.stdout
    print("This will also be flushed immediately.")
    sys.stdout.flush()

    # Extract the file name from the full file path
    filename = os.path.basename(file_path)

    # Create a socket for TCP communication
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))  # Connect to the server

    try:
        # Send the length of the file name as a 4-byte integer
        filename_length = len(filename)
        client_socket.sendall(filename_length.to_bytes(4, 'big'))

        # Send the actual file name
        client_socket.sendall(filename.encode())

        # Send the audio file
        with open(file_path, "rb") as f:
            while chunk := f.read(1024):
                client_socket.sendall(chunk)

        print("finished sending audio")
        client_socket.sendall(b"END")


        print(f"File '{filename}' sent to the server.")

        # Receive the result from the server
        result = client_socket.recv(1024).decode()
        print(f"Result from server: {result}")

    except Exception as e:
        # Handle any exceptions that occur during file transfer or communication
        print(f"Error occurred: {e}")
    finally:
        # Ensure the socket is closed after communication
        client_socket.close()

if __name__ == "__main__":
    # Allow the user to choose an audio file
    file = 'test_audio.wav'
    audio_file = choose_audio_file(file)
    print("Audio file selected: ", audio_file)

    # Send the selected file to the server
    send_audio_file(audio_file)
