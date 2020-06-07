#!/usr/bin/env python
from app import app
import StyleYourArt
import argparse

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)

    ## parse arguments for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="debug flask")
    args = vars(ap.parse_args())

    if args["debug"]:
        app.run(debug=True, port=8080)
    else:
       app.run(host='0.0.0.0', threaded=True ,port=8080)