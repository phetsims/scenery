/* eslint-disable */
import tile from './shared/tile.js';
import drawtag from './shared/drawtag.js';
import bump from './shared/bump.js';
import config from './shared/config.js';

export default `${config}
${bump}
${drawtag}
${tile}
@group(0)@binding(0)
var<uniform>_l:_aL;@group(0)@binding(1)
var<storage>_q:array<u32>;@group(0)@binding(2)
var<storage>_gT:array<vec4<f32>>;@group(0)@binding(3)
var<storage,read_write>_af:_em;@group(0)@binding(4)
var<storage,read_write>_J:array<_aJ>;@group(0)@binding(5)
var<storage,read_write>_t:array<_aI>;const _i=256u;var<workgroup>_aX:array<u32,_i>;var<workgroup>_iX:u32;@compute @workgroup_size(256)
fn main(
@builtin(global_invocation_id)_E:vec3<u32>,@builtin(local_invocation_id)_e:vec3<u32>,){let _ab=atomicLoad(&_af._ab);if(_ab&_el)!=0u{return;}let SX=1.0/f32(_cu);let SY=1.0/f32(_bg);let _aa=_E.x;var _at=_du;if _aa<_l._dx{_at=_q[_l._cJ+_aa];}var x0=0;var y0=0;var x1=0;var y1=0;if _at !=_du&&_at !=_ee{let _b=_gT[_aa];x0=i32(floor(_b.x*SX));y0=i32(floor(_b.y*SY));x1=i32(ceil(_b.z*SX));y1=i32(ceil(_b.w*SY));}let _ft=u32(clamp(x0,0,i32(_l._aB)));let _fs=u32(clamp(y0,0,i32(_l._cs)));let _fr=u32(clamp(x1,0,i32(_l._aB)));let _fq=u32(clamp(y1,0,i32(_l._cs)));let _W=(_fr-_ft)*(_fq-_fs);var _dP=_W;_aX[_e.x]=_W;for(var i=0u;i<firstTrailingBit(_i);i+=1u){workgroupBarrier();if _e.x>=(1u<<i){_dP+=_aX[_e.x-(1u<<i)];}workgroupBarrier();_aX[_e.x]=_dP;}if _e.x==_i-1u{let _j=_aX[_i-1u];var _d=atomicAdd(&_af._a,_j);if _d+_j>_l._ii{_d=0u;atomicOr(&_af._ab,_fj);}_J[_aa]._t=_d;}storageBarrier();let _es=_J[_aa|(_i-1u)]._t;storageBarrier();if _aa<_l._dx{let _gV=select(0u,_aX[_e.x-1u],_e.x>0u);let _b=vec4(_ft,_fs,_fr,_fq);let _c=_aJ(_b,_es+_gV);_J[_aa]=_c;}let _gU=_aX[_i-1u];for(var i=_e.x;i<_gU;i+=_i){_t[_es+i]=_aI(0,0u);}}`
