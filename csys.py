import flask as fl
import json
import numpy as np

class VisData():
    def __init__(self):
        self.x = np.zeros(3, dtype=np.float64)
        self.q = np.zeros(4, dtype=np.float64)
        self.q[0] = 1.
        self.inputs = np.zeros(4, dtype=np.float64)

    def update(self, x, q, u):
        self.x[:] = x
        self.q[:] = q
        self.inputs[:] = u

visData = VisData()
visApp = fl.Flask(__name__, static_url_path='/static')

@visApp.route("/")
def hello_world():
    return fl.render_template("index.html")

@visApp.route("/pose")
def pose():
    arr = []
    pos = list(visData.x.round(4))
    quat = list(visData.q.round(4))
    ctl = list(visData.inputs)
    arr.append({'id': 0, 'type': 0, 'pos': pos, 'quat': quat, 'ctl': ctl})
    return json.dumps(arr)

# dont spam the console
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

#simThread = threading.Thread(target=runSim, daemon=True)
if __name__=="__main__":
    #simThread.start( )
    visData.update( np.array( [0.21, 0.1, -0.21] ), np.array([0.8, 0.0, -.0, -0.4]), np.array([0, 0, 0, 0]) )
    visApp.run( )


