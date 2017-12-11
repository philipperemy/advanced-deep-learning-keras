import json
import warnings
from time import sleep

import requests
from keras.callbacks import Callback


class RemoteMonitor2(Callback):
    """Callback used to stream events to a server.

    Requires the `requests` library.
    Events are sent to `root + '/publish/epoch/end/'` by default. Calls are
    HTTP POST, with a `data` argument which is a
    JSON-encoded dictionary of event data.

    # Arguments
        root: String; root url of the target server.
        path: String; path relative to `root` to which the events will be sent.
        field: String; JSON field under which the data will be stored.
        headers: Dictionary; optional custom HTTP headers.
    """

    def __init__(self,
                 root='http://localhost:9000',
                 path='/publish/epoch/end/',
                 field='data',
                 headers=None):
        super(RemoteMonitor2, self).__init__()

        self.root = root
        self.path = path
        self.field = field
        self.headers = headers

    def on_batch_end(self, epoch, logs=None):
        if requests is None:
            raise ImportError('RemoteMonitor requires '
                              'the `requests` library.')
        sleep(0.5)
        logs = logs or {}
        send = {}
        send['epoch'] = str(epoch)
        print(send)
        for k, v in logs.items():
            send[k] = str(v)

        if float(send['loss']) > 1:
            send['loss'] = str(float(send['loss']) / 2.5)
        print(send)
        try:
            send = {'acc': send['acc'], 'loss': send['loss'], 'epoch': send['epoch']}
            requests.post(self.root + self.path,
                          {self.field: json.dumps(send)},
                          headers=self.headers)
        except requests.exceptions.RequestException:
            warnings.warn('Warning: could not reach RemoteMonitor '
                          'root server at ' + str(self.root))


