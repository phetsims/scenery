/* eslint-disable */
import tile from './shared/tile.js';
import config from './shared/config.js';

export default `${config}
${tile}
@group(0)@binding(0)
var<uniform>_l:_aL;@group(0)@binding(1)
var<storage>_J:array<_aJ>;@group(0)@binding(2)
var<storage,read_write>_t:array<_aI>;const _i=256u;var<workgroup>_gk:array<u32,_i>;var<workgroup>_cD:array<u32,_i>;var<workgroup>_gj:array<u32,_i>;@compute @workgroup_size(256)
fn main(
@builtin(global_invocation_id)_E:vec3<u32>,@builtin(local_invocation_id)_e:vec3<u32>,){let _aa=_E.x;var _bo=0u;if _aa<_l._dx{let _c=_J[_aa];_gk[_e.x]=_c._b.z-_c._b.x;_bo=_c._b.w-_c._b.y;_gj[_e.x]=_c._t;}_cD[_e.x]=_bo;
for(var i=0u;i<firstTrailingBit(_i);i+=1u){workgroupBarrier();if _e.x>=(1u<<i){_bo+=_cD[_e.x-(1u<<i)];}workgroupBarrier();_cD[_e.x]=_bo;}workgroupBarrier();let _hZ=_cD[_i-1u];for(var _ag=_e.x;_ag<_hZ;_ag+=_i){var _aj=0u;for(var i=0u;i<firstTrailingBit(_i);i+=1u){let _bp=_aj+((_i/2u)>>i);if _ag>=_cD[_bp-1u]{_aj=_bp;}}let _m=_gk[_aj];if _m>0u{var _dY=_ag-select(0u,_cD[_aj-1u],_aj>0u);var _ap=_gj[_aj]+_dY*_m;var _Z=_t[_ap]._w;for(var x=1u;x<_m;x+=1u){_ap+=1u;_Z+=_t[_ap]._w;_t[_ap]._w=_Z;}}}}`
