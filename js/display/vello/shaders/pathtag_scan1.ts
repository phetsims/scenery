/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';

export default `${config}
${pathtag}
@group(0)@binding(0)
var<storage>_ao:array<_ac>;@group(0)@binding(1)
var<storage>_gW:array<_ac>;@group(0)@binding(2)
var<storage,read_write>_bP:array<_ac>;const _bA=8u;const _i=256u;var<workgroup>_be:array<_ac,_i>;var<workgroup>_bz:array<_ac,_i>;@compute @workgroup_size(256)
fn main(
@builtin(global_invocation_id)_E:vec3<u32>,@builtin(local_invocation_id)_e:vec3<u32>,@builtin(workgroup_id)_ah:vec3<u32>,){var agg=_gE();if _e.x<_ah.x{agg=_gW[_e.x];}_be[_e.x]=agg;for(var i=0u;i<_bA;i+=1u){workgroupBarrier();if _e.x+(1u<<i)<_i{let _Y=_be[_e.x+(1u<<i)];agg=_by(agg,_Y);}workgroupBarrier();_be[_e.x]=agg;}let ix=_E.x;agg=_ao[ix];_bz[_e.x]=agg;for(var i=0u;i<_bA;i+=1u){workgroupBarrier();if _e.x>=1u<<i{let _Y=_bz[_e.x-(1u<<i)];agg=_by(_Y,agg);}workgroupBarrier();_bz[_e.x]=agg;}workgroupBarrier();
var tm=_be[0];if _e.x>0u{tm=_by(tm,_bz[_e.x-1u]);}_bP[ix]=tm;}`
