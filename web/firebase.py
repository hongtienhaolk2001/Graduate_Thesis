import pyrebase
"""
project name
project id    : long-equinox-405504
project number: 102318358773
"""

config = {
    "apiKey": "AIzaSyBg_9QTfbP9616v7KshDbywUmd0eMBqjhg",
    "authDomain": "fir-ca79b.firebaseapp.com",
    "databaseURL": "https://fir-ca79b-default-rtdb.asia-southeast1.firebasedatabase.app",
    "projectId": "fir-ca79b",
    'storageBucket': "fir-ca79b.appspot.com",
    'messagingSenderId': "703924362976",
    'appId': "1:703924362976:web:067ac1a24480bfb4647a85",
    'measurementId': "G-D68182712S"
}


def connect_firebase():
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    return db