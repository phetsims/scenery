/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';

export default `${config}
${pathtag}
@group(0)@binding(0)var<uniform>_m:_aG;@group(0)@binding(1)var<storage>_n:array<u32>;@group(0)@binding(2)var<storage>_aq:array<_ad>;@group(0)@binding(3)var<storage,read_write>_bV:array<_ad>;const _bH=8u;const _j=256u;var<workgroup>_bG:array<_ad,_j>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)_G:vec3<u32>,@builtin(local_invocation_id)_f:vec3<u32>,@builtin(workgroup_id)_aj:vec3<u32>,){let ix=_G.x;let _D=_n[_m._dE+ix];var _dK=_el(_D);_bG[_f.x]=_dK;for(var i=0u;i<_bH;i+=1u){workgroupBarrier();if _f.x>=1u<<i{let _Z=_bG[_f.x-(1u<<i)];_dK=_bF(_Z,_dK);}workgroupBarrier();_bG[_f.x]=_dK;}workgroupBarrier();var tm=_aq[_aj.x];if _f.x>0u{tm=_bF(tm,_bG[_f.x- 1u]);}_bV[ix]=tm;}`
