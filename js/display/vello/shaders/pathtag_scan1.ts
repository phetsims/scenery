/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';

export default `${config}
${pathtag}
@group(0)@binding(0)var<storage>_aq:array<_ad>;@group(0)@binding(1)var<storage>_he:array<_ad>;@group(0)@binding(2)var<storage,read_write>_bV:array<_ad>;const _bH=8u;const _j=256u;var<workgroup>_bh:array<_ad,_j>;var<workgroup>_bG:array<_ad,_j>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)_G:vec3<u32>,@builtin(local_invocation_id)_f:vec3<u32>,@builtin(workgroup_id)_aj:vec3<u32>,){var agg=_gM();if _f.x<_aj.x{agg=_he[_f.x];}_bh[_f.x]=agg;for(var i=0u;i<_bH;i+=1u){workgroupBarrier();if _f.x+(1u<<i)<_j{let _Z=_bh[_f.x+(1u<<i)];agg=_bF(agg,_Z);}workgroupBarrier();_bh[_f.x]=agg;}let ix=_G.x;agg=_aq[ix];_bG[_f.x]=agg;for(var i=0u;i<_bH;i+=1u){workgroupBarrier();if _f.x>=1u<<i{let _Z=_bG[_f.x-(1u<<i)];agg=_bF(_Z,agg);}workgroupBarrier();_bG[_f.x]=agg;}workgroupBarrier();var tm=_bh[0];if _f.x>0u{tm=_bF(tm,_bG[_f.x- 1u]);}_bV[ix]=tm;}`
