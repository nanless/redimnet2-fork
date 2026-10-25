"""等待同一控制面观测的稳定新材料；不伪造状态、不初始化CUDA。"""
import json
import time
import strict_restore as S


def wait_for_terminal(timeout=1800):
    start = time.monotonic(); last = None
    while True:
        try:
            refs, evidence = S.fresh_terminals()
            return {'refs': refs, 'evidence': evidence}
        except (OSError, ValueError) as error:
            text = str(error)
            if last != text:
                print('WAITING_FRESH_TERMINAL ' + json.dumps({'reason': text, 'maximum_seconds': timeout}), flush=True)
                last = text
            S.need(time.monotonic() - start < timeout, 'fresh terminal evidence wait exhausted; no CUDA launched')
            time.sleep(min(1., max(.01, timeout - (time.monotonic() - start))))


def verify_unchanged(expected):
    refs, evidence = S.fresh_terminals()
    S.need(refs == expected['refs'], 'fresh terminal bytes changed under admission; repeat entire admission')
    return evidence
