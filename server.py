#!flask/bin/python
###################################################
# ICAO Check
###################################################

from flask import Flask, jsonify, request, send_from_directory
import base64
import json
# import requests  # <-- I don't know why this is here, but I leave it for now
import imghdr
import tempfile
from subprocess32 import check_output, CalledProcessError


app = Flask(__name__)


@app.route('/', methods=['GET'])
@app.route('/<path:path>', methods=['GET'])
def send_page(path='/'):
    return send_from_directory('./', 'index.html')


@app.route('/api/icaochecker', methods=['POST'])
def ad_icaocheck():
    # check if json request and all needed data included
    if not request.json or \
       not ('clientQueryId' in request.json) or \
       not ('imageData' in request.json):
        print("ERROR: cannot parse JSON object")
        return jsonify({'Error': 'cannot parse JSON object'}), 400

    if 'documentImage' not in request.json['imageData']:
        print("MESSAGE: documentImage is not defined")
        return jsonify({'message': 'documentImage is not defined'}), 400

    # get b64 encoded image from json request
    img_base64 = base64.b64decode(request.json['imageData']['documentImage'])

    # get autorotate boolean; assume false, if not present
    autorotate = 'autorotate' in request.json and \
                 type(request.json['autorotate']) is bool and \
                 request.json['autorotate']

    # image needs to be jpeg or png
    fmt = imghdr.what('foo', img_base64)
    if not (fmt == 'jpeg' or fmt == 'png'):
        return jsonify({'message': 'false image file format, jpg or png file is expected'}), 400
    # print("autorotate is " + ("on" if autorotate else "off"))

    # dump image to file
    # Idea: In future just give base64 data directly to subprocesses,
    #       so they don't need to load the files.
    f = tempfile.NamedTemporaryFile(mode='wb')
    f.write(img_base64)
    f.seek(0)

    output = dict()

    ###################################################
    # Run AUTOROTATION
    #     by David Döring
    ###################################################
    if autorotate:
        try:
            o = check_output(
                ['./geometry_check/icao-venv/bin/python3',
                 './geometry_check/autorotate.py',
                 f.name],
                timeout=180
            )
            odct = json.loads(o)
            editdct = dict(("autorotate_" + key, value)
                           for (key, value) in odct.items())
            output.update(editdct)
        except CalledProcessError:
            output['correctly_executed'] = False
            output['Error'] = output.get(
                'Error', '') + 'error executing autorotate;'
        except ValueError:
            output['correctly_executed'] = False
            output['Error'] = output.get(
                'Error', '') + 'JSON parsing error autorotate;'

    ###################################################
    # Run GEOMETRY CHECK
    #     by David Döring
    ###################################################
    try:
        o = check_output(
            ['./geometry_check/icao-venv/bin/python3',
             './geometry_check/geometry_check.py',
             f.name],
            timeout=180
        )
        odct = json.loads(o)
        editdct = dict(("geometry_" + key, value)
                       for (key, value) in odct.items())
        output.update(editdct)
    except CalledProcessError:
        output['correctly_executed'] = False
        output['Error'] = output.get(
            'Error', '') + 'error executing geometry_check;'
    except ValueError:
        output['correctly_executed'] = False
        output['Error'] = output.get(
            'Error', '') + 'JSON parsing error geometry_check;'

    ###################################################
    # Run NEUTRAL FACE CHECK
    #     by Oliver Keil
    ###################################################
    try:
        o = check_output(
            ['./neutralface/neutralface.run',
             './data/shape_predictor_68_face_landmarks.dat',
             f.name],
            timeout=180
        )
        odct = json.loads(o)
        editdct = dict(("neutralface_" + key, value)
                       for (key, value) in odct.items())
        output.update(editdct)
    except CalledProcessError as e:
        output['correctly_executed'] = False
        output['Error'] = output.get(
            'Error', '') + 'error executing neutral_face;'
        output['Error_msg'] = output.get('Error_msg', '') + e.output.decode()
    except ValueError:
        output['correctly_executed'] = False
        output['Error'] = output.get(
            'Error', '') + 'JSON parsing error neutral_face;'
        output['Error_msg'] = output.get('Error_msg', '') + o

    f.close()

    # jsonify output and send back to requester
    return jsonify(output), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
