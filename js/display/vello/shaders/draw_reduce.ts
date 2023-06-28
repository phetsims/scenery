/* eslint-disable */
import drawtag from './shared/drawtag.js';
import config from './shared/config.js';

export default `${config}
${drawtag}
@group(0)@binding(0)
var<uniform>_l:_aL;@group(0)@binding(1)
var<storage>_q:array<u32>;@group(0)@binding(2)
var<storage,read_write>_ao:array<_aE>;const _i=256u;var<workgroup>_as:array<_aE,_i>;@compute @workgroup_size(256)
fn main(
@builtin(global_invocation_id)_E:vec3<u32>,@builtin(local_invocation_id)_e:vec3<u32>,){let ix=_E.x;let _D=_q[_l._cI+ix];var agg=_gE(_D);_as[_e.x]=agg;for(var i=0u;i<firstTrailingBit(_i);i+=1u){workgroupBarrier();if _e.x+(1u<<i)<_i{let _Y=_as[_e.x+(1u<<i)];agg=_eh(agg,_Y);}workgroupBarrier();_as[_e.x]=agg;}if _e.x==0u{_ao[ix>>firstTrailingBit(_i)]=agg;}}`
