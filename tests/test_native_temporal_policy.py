"""CPU contract tests; no native extension or GPU required."""
import dataclasses,importlib.util,sys
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[1]/'comfy_kitchen/temporal_attention.py'
spec=importlib.util.spec_from_file_location('kitchen_temporal_policy_test',p)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)

def valid(**changes):
 args=dict(source_start=4,source_stop=10,target_start=12,target_stop=21,source_frame_rows=2,target_frame_rows=3,frame_times=(0.,.7,1.4))
 args.update(changes);return module.TemporalPolicy(**args)

def test_native_policy_is_immutable_and_accepts_real_timestamps():
 p=valid();assert p.frame_times==(0.,.7,1.4)
 with pytest.raises(dataclasses.FrozenInstanceError):p.strength=3.
 assert p._buffers=={}

@pytest.mark.parametrize('changes',[
 {'source_start':True},{'source_start':-1},{'source_stop':13},{'target_start':8},
 {'target_stop':2**24},{'source_frame_rows':0},{'target_frame_rows':4},
 {'frame_times':()},{'frame_times':(0.,0.,1.)},{'frame_times':(0.,float('nan'),1.)},
 {'strength':-1.},{'cap':-1.},{'window':-1.},{'falloff':0.},{'cap':float('inf')},
])
def test_bad_policy_is_rejected(changes):
 with pytest.raises(ValueError):valid(**changes)

def test_zero_penalty_and_one_frame_are_valid():
 valid(strength=0.,cap=0.)
 p=module.TemporalPolicy(0,2,4,7,2,3,(0.,))
 assert len(p.frame_times)==1
