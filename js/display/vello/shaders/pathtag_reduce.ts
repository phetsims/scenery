/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';

export default `${config}
${pathtag}
@group(0)@binding(0)
var<uniform>_l:_aL;@group(0)@binding(1)
var<storage>_q:array<u32>;@group(0)@binding(2)
var<storage,read_write>_ao:array<_ac>;const _bA=8u;const _i=256u;var<workgroup>_as:array<_ac,_i>;@compute @workgroup_size(256)
fn main(
@builtin(global_invocation_id)_E:vec3<u32>,@builtin(local_invocation_id)_e:vec3<u32>,){let ix=_E.x;let _D=_q[_l._du+ix];var agg=_ec(_D);_as[_e.x]=agg;for(var i=0u;i<firstTrailingBit(_i);i+=1u){workgroupBarrier();if _e.x+(1u<<i)<_i{let _Y=_as[_e.x+(1u<<i)];agg=_by(agg,_Y);}workgroupBarrier();_as[_e.x]=agg;}if _e.x==0u{_ao[ix>>_bA]=agg;}}`
