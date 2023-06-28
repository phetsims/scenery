/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';

export default `${config}
${pathtag}
@group(0)@binding(0)
var<storage>_gW:array<_ac>;@group(0)@binding(1)
var<storage,read_write>_ao:array<_ac>;const _bA=8u;const _i=256u;var<workgroup>_as:array<_ac,_i>;@compute @workgroup_size(256)
fn main(
@builtin(global_invocation_id)_E:vec3<u32>,@builtin(local_invocation_id)_e:vec3<u32>,){let ix=_E.x;var agg=_gW[ix];_as[_e.x]=agg;for(var i=0u;i<firstTrailingBit(_i);i+=1u){workgroupBarrier();if _e.x+(1u<<i)<_i{let _Y=_as[_e.x+(1u<<i)];agg=_by(agg,_Y);}workgroupBarrier();_as[_e.x]=agg;}if _e.x==0u{_ao[ix>>_bA]=agg;}}`
