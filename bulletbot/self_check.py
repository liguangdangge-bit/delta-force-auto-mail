"""Read-only packaged smoke check: no game input, authorization, or PAK operations."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


@patch('requests.sessions.Session.request', side_effect=AssertionError('Local client must not use HTTP'))
@patch('urllib.request.urlopen', side_effect=AssertionError('Local client must not use HTTP'))
def run(destination, urlopen, http):
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    os.environ['DELTAFORCE_BULLETBOT_FORCE_CPU'] = '1'
    from bulletbot.ocr.favorites_ocr import preload_onnx_runtime
    preload_onnx_runtime()
    from PyQt5.QtWidgets import QApplication
    from bulletbot.ui.mail_window import MailWindow
    from bulletbot.orchestration import MailIntegrationRuntime
    from bulletbot.reporting.run_recorder import RunRecorder
    from bulletbot.mail_storage.vision import TemplateCatalog, OcrEngine
    from bulletbot.ocr.numeric_ocr import NumericOcrEngine
    from bulletbot.ocr.local_models import general_model_params
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    app = QApplication([])
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        window = MailWindow(MailIntegrationRuntime(root), RunRecorder(root/'logs'), interactive=False)
        app.processEvents()
        assert window.centralWidget().objectName() == 'appRoot'
        window.close()
        app.processEvents()
    with patch('socket.create_connection', side_effect=AssertionError('Network disabled during self-check')):
        img = Image.new('RGB', (520, 110), 'white')
        ImageDraw.Draw(img).text((15,15), '12345', font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',52),fill='black')
        frame = np.asarray(img)[:,:,::-1].copy()
        words = OcrEngine().detect(frame)
        number = NumericOcrEngine().recognize_line(frame)
    assert any('12345' in w.text for w in words), words
    assert number and '12345' in number[0], number
    http.assert_not_called()
    urlopen.assert_not_called()
    result = {'status':'passed','ui':'mail-only shell constructed and closed',
              'templates':len(TemplateCatalog().specs),'models':general_model_params(),
              'general_ocr':[w.text for w in words],'numeric_ocr':number[0],
              'game_operations':False, 'http_requests':0, 'local_only':True}
    Path(destination).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
