/* eslint-disable */
import drawtag from './shared/drawtag.js';
import clip from './shared/clip.js';
import bbox from './shared/bbox.js';
import config from './shared/config.js';

export default `${config}
${bbox}
${clip}
${drawtag}
@group(0)@binding(0)
var<uniform>_l:_aL;@group(0)@binding(1)
var<storage>_cA:array<_ek>;@group(0)@binding(2)
var<storage>_bw:array<_bY>;@group(0)@binding(3)
var<storage>_ao:array<_aS>;@group(0)@binding(4)
var<storage>_hQ:array<_fi>;@group(0)@binding(5)
var<storage,read_write>_bM:array<_aE>;@group(0)@binding(6)
var<storage,read_write>_hP:array<vec4<f32>>;const _i=256u;var<workgroup>_aD:array<_aS,510>;var<workgroup>_bU:array<u32,_i>;var<workgroup>_dj:array<vec4<f32>,_i>;var<workgroup>_dU:array<vec4<f32>,_i>;var<workgroup>_dT:array<i32,_i>;fn _hS(_n:ptr<function,_aS>,_hO:u32)->i32{var ix=_hO;var j=0u;while j<firstTrailingBit(_i){let _p=2u*_i-(2u<<(firstTrailingBit(_i)-j));if((ix>>j)&1u)!=0u{let _dk=_dz(_aD[_p+(ix>>j)-1u],*_n);if _dk.b>0u{break;}*_n=_dk;ix-=1u<<j;}j+=1u;}if ix>0u{while j>0u{j-=1u;let _p=2u*_i-(2u<<(firstTrailingBit(_i)-j));let _dk=_dz(_aD[_p+(ix>>j)-1u],*_n);if _dk.b==0u{*_n=_dk;ix-=1u<<j;}}}if ix>0u{return i32(ix)-1;}else{return i32(~0u-(*_n).a);}}fn _hR(ix:u32)->i32{if ix<_l._dw{return _cA[ix]._N;}else{return-2147483648;}}@compute @workgroup_size(256)
fn main(
@builtin(global_invocation_id)_E:vec3<u32>,@builtin(local_invocation_id)_e:vec3<u32>,@builtin(workgroup_id)_ah:vec3<u32>,){var _n:_aS;if _e.x<_ah.x{_n=_ao[_e.x];}_aD[_e.x]=_n;for(var i=0u;i<firstTrailingBit(_i);i+=1u){workgroupBarrier();if _e.x+(1u<<i)<_i{let _Y=_aD[_e.x+(1u<<i)];_n=_dz(_n,_Y);}workgroupBarrier();_aD[_e.x]=_n;}workgroupBarrier();let _eR=_aD[0].b;

let sp=_i-1u-_e.x;var ix=0u;for(var i=0u;i<firstTrailingBit(_i);i+=1u){let _bp=ix+((_i/2u)>>i);if sp<_aD[_bp].b{ix=_bp;}}let b=_aD[ix].b;var _b=vec4(-1e9,-1e9,1e9,1e9);if sp<b{let el=_hQ[ix*_i+b-sp-1u];_bU[_e.x]=el._dy;_b=el._b;}for(var i=0u;i<firstTrailingBit(_i);i+=1u){_dj[_e.x]=_b;workgroupBarrier();if _e.x>=(1u<<i){_b=_er(_dj[_e.x-(1u<<i)],_b);}workgroupBarrier();}_dj[_e.x]=_b;
let _aH=_hR(_E.x);let _bV=_aH>=0;_n=_aS(1u-u32(_bV),u32(_bV));_aD[_e.x]=_n;if _bV{let _T=_bw[_aH];_b=vec4(f32(_T.x0),f32(_T.y0),f32(_T.x1),f32(_T.y1));}else{_b=vec4(-1e9,-1e9,1e9,1e9);}var _fY=0u;for(var i=0u;i<firstTrailingBit(_i)-1u;i+=1u){let _gb=2u*_i-(1u<<(firstTrailingBit(_i)-i));workgroupBarrier();if _e.x<1u<<(firstTrailingBit(_i)-1u-i){let _ga=_fY+_e.x*2u;_aD[_gb+_e.x]=_dz(_aD[_ga],_aD[_ga+1u]);}_fY=_gb;}workgroupBarrier();
_n=_aS();var _X=_hS(&_n,_e.x);_dT[_e.x]=_X;workgroupBarrier();let _dV=select(_X-1,_dT[_X],_X>=0);var _P:i32;if _X>=0{_P=i32(_ah.x*_i)+_X;}else if _X+i32(_eR)>=0{_P=i32(_bU[i32(_i)+_X]);}else{_P=-1;}for(var i=0u;i<firstTrailingBit(_i);i+=1u){if i !=0u{_dT[_e.x]=_X;}_dU[_e.x]=_b;workgroupBarrier();if _X>=0{_b=_er(_dU[_X],_b);_X=_dT[_X];}workgroupBarrier();}if _X+i32(_eR)>=0{_b=_er(_dj[i32(_i)+_X],_b);}_dU[_e.x]=_b;workgroupBarrier();if !_bV&&_E.x<_l._dw{let _fZ=_cA[_P];let _N=_fZ._N;let _dy=_fZ.ix;let ix=~_aH;_bM[ix]._N=u32(_N);
_bM[ix]._bq=_bM[_dy]._bq;if _dV>=0{_b=_dU[_dV];}else if _dV+i32(_eR)>=0{_b=_dj[i32(_i)+_dV];}else{_b=vec4(-1e9,-1e9,1e9,1e9);}}if _E.x<_l._dw{_hP[_E.x]=_b;}}`
