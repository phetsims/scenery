/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';

export default `${config}
${pathtag}
@group(0)@binding(0)var<uniform>_m:_aG;@group(0)@binding(1)var<storage>_n:array<u32>;@group(0)@binding(2)var<storage,read_write>_aq:array<_ad>;const _bH=8u;const _j=256u;var<workgroup>_at:array<_ad,_j>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)_G:vec3<u32>,@builtin(local_invocation_id)_f:vec3<u32>,){let ix=_G.x;let _D=_n[_m._dE+ix];var agg=_el(_D);_at[_f.x]=agg;for(var i=0u;i<firstTrailingBit(_j);i+=1u){workgroupBarrier();if _f.x+(1u<<i)<_j{let _Z=_at[_f.x+(1u<<i)];agg=_bF(agg,_Z);}workgroupBarrier();_at[_f.x]=agg;}if _f.x==0u{_aq[ix>>_bH]=agg;}}`
