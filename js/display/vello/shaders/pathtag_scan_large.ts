/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';

export default `${config}
${pathtag}
@group(0)@binding(0)
var<uniform>_l:_aL;@group(0)@binding(1)
var<storage>_q:array<u32>;@group(0)@binding(2)
var<storage>_ao:array<_ac>;@group(0)@binding(3)
var<storage,read_write>_bO:array<_ac>;const _bA=8u;const _i=256u;var<workgroup>_bz:array<_ac,_i>;@compute @workgroup_size(256)
fn main(
@builtin(global_invocation_id)_E:vec3<u32>,@builtin(local_invocation_id)_e:vec3<u32>,@builtin(workgroup_id)_ah:vec3<u32>,){let ix=_E.x;let _D=_q[_l._du+ix];var _dB=_ec(_D);_bz[_e.x]=_dB;for(var i=0u;i<_bA;i+=1u){workgroupBarrier();if _e.x>=1u<<i{let _Y=_bz[_e.x-(1u<<i)];_dB=_by(_Y,_dB);}workgroupBarrier();_bz[_e.x]=_dB;}workgroupBarrier();
var tm=_ao[_ah.x];if _e.x>0u{tm=_by(tm,_bz[_e.x-1u]);}_bO[ix]=tm;}`
