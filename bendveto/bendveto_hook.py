"""lf-009 driver wrapper with the bend veto (bendveto.py). Weights, route handling and controller are unchanged; the hook
only replaces the returned plan when the veto applies.
BENDVETO_ARM = off (parity) | shadow (log only) | active. Logs one BENDVETO line per planning call."""
import json
import logging
import os
import runpy
import shlex
import sys
import threading

import numpy as np
import torch
from py123d_garage.evaluation.alpasim.help.driver_service import GarageDriver
from py123d_garage.policy.transfuser.transfuser_policy import TransfuserPolicy
from bendveto import bend_veto

MODE = os.environ.get('BENDVETO_ARM', 'active')
assert MODE in ('off', 'shadow', 'active')
SURFACE = (1, 3, 5)                      # road, centerline, vehicle
LOG = logging.getLogger('bendveto')
CONTEXT = threading.local()
CALLS = {}
ORIGINAL_FORWARD = TransfuserPolicy.forward
ORIGINAL_PLAN = GarageDriver._plan


def _reset():
    CONTEXT.capture = False
    CONTEXT.p_surf = CONTEXT.boxes = CONTEXT.default = None


def forward(self, features, navigation):
    prediction = ORIGINAL_FORWARD(self, features, navigation)
    if getattr(CONTEXT, 'capture', False):
        with torch.no_grad():
            p = torch.softmax(prediction.bev_semantic[0].float(), dim=0)
            CONTEXT.p_surf = p[list(SURFACE)].sum(0).cpu().numpy()
        CONTEXT.boxes = np.asarray(prediction.boxes.pred_bounding_box_vehicle_system[0]).copy()
        CONTEXT.default = prediction.ego_trajectory_se2[0].detach().float().cpu().numpy()
    return prediction


def plan(self, policy, session, session_uuid):
    _reset()
    CONTEXT.capture = MODE != 'off'
    try:
        nominal = ORIGINAL_PLAN(self, policy, session, session_uuid)
        if MODE == 'off':
            return nominal
        k = CALLS.get(session_uuid, 0); CALLS[session_uuid] = k + 1
        info = dict(session=session_uuid, mode=MODE)
        out = nominal
        try:
            nom = np.asarray(nominal, float)
            if CONTEXT.p_surf is None or CONTEXT.default is None:
                info.update(applied=False, reason='missing_capture')
            elif nom.shape != CONTEXT.default.shape or np.abs(nom - CONTEXT.default).max() > 1e-3:
                info.update(applied=False, reason='frame_mismatch')
            else:
                cand, vinfo = bend_veto(nom, CONTEXT.p_surf, CONTEXT.boxes, k)
                info.update(vinfo)
                if MODE == 'active' and vinfo.get('applied'):
                    out = cand.astype(np.asarray(nominal).dtype)
        except Exception as e:  # any failure keeps the model's plan
            info.update(applied=False, reason='error', error=str(e)[:120]); out = nominal
        LOG.info('BENDVETO %s', json.dumps(info))
        return out
    finally:
        _reset()


GarageDriver._plan = plan
TransfuserPolicy.forward = forward
sys.argv = ['evaluate', *shlex.split(os.environ['PY123D_GARAGE_CONFIG']),
            'host=' + os.environ.get('ALPASIM_DRIVER_HOST', '0.0.0.0'),
            'port=' + os.environ.get('ALPASIM_DRIVER_PORT', '6789')]
runpy.run_module('py123d_garage.evaluation.alpasim.evaluate', run_name='__main__')
