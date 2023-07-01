/* eslint-disable */
import tile from './shared/tile.js';
import config from './shared/config.js';

export default `${config}
${tile}
@group(0)@binding(0)var<uniform>_m:_aG;@group(0)@binding(1)var<storage>_M:array<_aL>;@group(0)@binding(2)var<storage,read_write>_w:array<_aK>;const _j=256u;var<workgroup>_gq:array<u32,_j>;var<workgroup>_cL:array<u32,_j>;var<workgroup>_gp:array<u32,_j>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)_G:vec3<u32>,@builtin(local_invocation_id)_f:vec3<u32>,){let _ab=_G.x;var _bt=0u;if _ab<_m._cz{let _c=_M[_ab];_gq[_f.x]=_c._b.z-_c._b.x;_bt=_c._b.w-_c._b.y;_gp[_f.x]=_c._w;}_cL[_f.x]=_bt;for(var i=0u;i<firstTrailingBit(_j);i+=1u){workgroupBarrier();if _f.x>=(1u<<i){_bt+=_cL[_f.x-(1u<<i)];}workgroupBarrier();_cL[_f.x]=_bt;}workgroupBarrier();let _ig=_cL[_j- 1u];for(var _ai=_f.x;_ai<_ig;_ai+=_j){var _al=0u;for(var i=0u;i<firstTrailingBit(_j);i+=1u){let _bu=_al+((_j/2u)>>i);if _ai>=_cL[_bu- 1u]{_al=_bu;}}let _p=_gq[_al];if _p>0u{var _eg=_ai-select(0u,_cL[_al- 1u],_al>0u);var _ar=_gp[_al]+_eg*_p;var _aa=_w[_ar]._B;for(var x=1u;x<_p;x+=1u){_ar+=1u;_aa+=_w[_ar]._B;_w[_ar]._B=_aa;}}}}`
