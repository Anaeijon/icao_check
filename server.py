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

    # get autotransform boolean; assume false, if not present
    autotransform = 'autotransform' in request.json and \
        type(request.json['autotransform']) is bool and \
        request.json['autotransform']

    # image needs to be jpeg or png
    fmt = imghdr.what('foo', img_base64)
    if not (fmt == 'jpeg' or fmt == 'png'):
        return jsonify({'message': 'false image file format, jpg or png file is expected'}), 400
    # print("autotransform is " + ("on" if autotransform else "off"))

    # dump image to file
    # Idea: In future just give base64 data directly to subprocesses,
    #       so they don't need to load the files.
    f = tempfile.NamedTemporaryFile(mode='wb', suffix="." + fmt)
    f.write(img_base64)
    f.seek(0)

    # output_checks dict will be returned to client
    output_checks = dict()
    # check_output['imageData'] = dict({'documentImage': base64.b64encode(img_base64)})

    ###################################################
    # Run AUTOROTATION
    #     by David Doering
    ###################################################
    # WARNING: THIS WILL OVERWRITE THE FILE, IF TRANSFORM POSSIBLE
    if autotransform:
        try:
            o = check_output(
                ['./geometry_check/icao-venv/bin/python3',
                 './geometry_check/autotransform.py',
                 f.name,
                 f.name],
                timeout=180
            )
            odct = json.loads(o)
            editdct = dict(("autotransform_" + key, value)
                           for (key, value) in odct.items())
            output_checks.update(editdct)
        except CalledProcessError:
            output_checks['correctly_executed'] = False
            output_checks['Error'] = output_checks.get(
                'Error', '') + 'error executing autotransform;'
        except ValueError:
            output_checks['correctly_executed'] = False
            output_checks['Error'] = output_checks.get(
                'Error', '') + 'JSON parsing error autotransform;'

    ###################################################
    # Run GEOMETRY CHECK
    #     by David Doering
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
        output_checks.update(editdct)
    except CalledProcessError:
        output_checks['correctly_executed'] = False
        output_checks['Error'] = output_checks.get(
            'Error', '') + 'error executing geometry_check;'
    except ValueError:
        output_checks['correctly_executed'] = False
        output_checks['Error'] = output_checks.get(
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
        output_checks.update(editdct)
    except CalledProcessError as e:
        output_checks['correctly_executed'] = False
        output_checks['Error'] = output_checks.get(
            'Error', '') + 'error executing neutral_face;'
        output_checks['Error_msg'] = output_checks.get(
            'Error_msg', '') + e.output.decode()
    except ValueError:
        output_checks['correctly_executed'] = False
        output_checks['Error'] = output_checks.get(
            'Error', '') + 'JSON parsing error neutral_face;'
        output_checks['Error_msg'] = output_checks.get('Error_msg', '') + o

    with open(f.name, "rb") as f:
        f.seek(0)
        b64_output_data = f.read()
    f.close()

    # jsonify output_checks and send back to requester
    return jsonify({'checks': output_checks,
                    'imageData': {
                        'documentImage': base64.b64encode(b64_output_data)
                    }}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
