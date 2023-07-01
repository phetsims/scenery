/* eslint-disable */
import tile from './shared/tile.js';
import drawtag from './shared/drawtag.js';
import bump from './shared/bump.js';
import config from './shared/config.js';

export default `${config}
${bump}
${drawtag}
${tile}
@group(0)@binding(0)var<uniform>_m:_aG;@group(0)@binding(1)var<storage>_n:array<u32>;@group(0)@binding(2)var<storage>_hb:array<vec4<f32>>;@group(0)@binding(3)var<storage,read_write>_ah:_eu;@group(0)@binding(4)var<storage,read_write>_M:array<_aL>;@group(0)@binding(5)var<storage,read_write>_w:array<_aK>;const _j=256u;var<workgroup>_aZ:array<u32,_j>;var<workgroup>_je:u32;var<workgroup>_fv:u32;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)_G:vec3<u32>,@builtin(local_invocation_id)_f:vec3<u32>,){if _f.x==0u{_fv=atomicLoad(&_ah._ac);}let _ac=workgroupUniformLoad(&_fv);if(_ac&_et)!=0u{return;}let SX=1./f32(_cC);let SY=1./f32(_bk);let _ab=_G.x;var _au=_cy;if _ab<_m._cz{_au=_n[_m._cS+_ab];}var x0=0;var y0=0;var x1=0;var y1=0;if _au!=_cy&&_au!=_em{let _b=_hb[_ab];if _b.x<_b.z&&_b.y<_b.w{x0=i32(floor(_b.x*SX));y0=i32(floor(_b.y*SY));x1=i32(ceil(_b.z*SX));y1=i32(ceil(_b.w*SY));}}let _fz=u32(clamp(x0,0,i32(_m._aC)));let _fy=u32(clamp(y0,0,i32(_m._cA)));let _fx=u32(clamp(x1,0,i32(_m._aC)));let _fw=u32(clamp(y1,0,i32(_m._cA)));let _X=(_fx-_fz)*(_fw-_fy);var _dX=_X;_aZ[_f.x]=_X;for(var i=0u;i<firstTrailingBit(_j);i+=1u){workgroupBarrier();if _f.x>=(1u<<i){_dX+=_aZ[_f.x-(1u<<i)];}workgroupBarrier();_aZ[_f.x]=_dX;}if _f.x==_j- 1u{let _k=_aZ[_j- 1u];var _d=atomicAdd(&_ah._a,_k);if _d+_k>_m._ip{_d=0u;atomicOr(&_ah._ac,_fp);}_M[_ab]._w=_d;}storageBarrier();let _ez=_M[_ab|(_j- 1u)]._w;storageBarrier();if _ab<_m._cz{let _hd=select(0u,_aZ[_f.x- 1u],_f.x>0u);let _b=vec4(_fz,_fy,_fx,_fw);let _c=_aL(_b,_ez+_hd);_M[_ab]=_c;}let _hc=_aZ[_j- 1u];for(var i=_f.x;i<_hc;i+=_j){_w[_ez+i]=_aK(0,0u);}}`
