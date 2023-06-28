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
var<storage>_cz:array<_ej>;@group(0)@binding(2)
var<storage>_bw:array<_bX>;@group(0)@binding(3)
var<storage>_ao:array<_aS>;@group(0)@binding(4)
var<storage>_hO:array<_fh>;@group(0)@binding(5)
var<storage,read_write>_bL:array<_aE>;@group(0)@binding(6)
var<storage,read_write>_hN:array<vec4<f32>>;const _i=256u;var<workgroup>_aD:array<_aS,510>;var<workgroup>_bT:array<u32,_i>;var<workgroup>_di:array<vec4<f32>,_i>;var<workgroup>_dT:array<vec4<f32>,_i>;var<workgroup>_dS:array<i32,_i>;fn _hQ(_n:ptr<function,_aS>,_hM:u32)->i32{var ix=_hM;var j=0u;while j<firstTrailingBit(_i){let _p=2u*_i-(2u<<(firstTrailingBit(_i)-j));if((ix>>j)&1u)!=0u{let _dj=_dy(_aD[_p+(ix>>j)-1u],*_n);if _dj.b>0u{break;}*_n=_dj;ix-=1u<<j;}j+=1u;}if ix>0u{while j>0u{j-=1u;let _p=2u*_i-(2u<<(firstTrailingBit(_i)-j));let _dj=_dy(_aD[_p+(ix>>j)-1u],*_n);if _dj.b==0u{*_n=_dj;ix-=1u<<j;}}}if ix>0u{return i32(ix)-1;}else{return i32(~0u-(*_n).a);}}fn _hP(ix:u32)->i32{if ix<_l._dv{return _cz[ix]._N;}else{return-2147483648;}}@compute @workgroup_size(256)
fn main(
@builtin(global_invocation_id)_E:vec3<u32>,@builtin(local_invocation_id)_e:vec3<u32>,@builtin(workgroup_id)_ah:vec3<u32>,){var _n:_aS;if _e.x<_ah.x{_n=_ao[_e.x];}_aD[_e.x]=_n;for(var i=0u;i<firstTrailingBit(_i);i+=1u){workgroupBarrier();if _e.x+(1u<<i)<_i{let _Y=_aD[_e.x+(1u<<i)];_n=_dy(_n,_Y);}workgroupBarrier();_aD[_e.x]=_n;}workgroupBarrier();let _eQ=_aD[0].b;

let sp=_i-1u-_e.x;var ix=0u;for(var i=0u;i<firstTrailingBit(_i);i+=1u){let _bp=ix+((_i/2u)>>i);if sp<_aD[_bp].b{ix=_bp;}}let b=_aD[ix].b;var _b=vec4(-1e9,-1e9,1e9,1e9);if sp<b{let el=_hO[ix*_i+b-sp-1u];_bT[_e.x]=el._dx;_b=el._b;}for(var i=0u;i<firstTrailingBit(_i);i+=1u){_di[_e.x]=_b;workgroupBarrier();if _e.x>=(1u<<i){_b=_eq(_di[_e.x-(1u<<i)],_b);}workgroupBarrier();}_di[_e.x]=_b;
let _aH=_hP(_E.x);let _bU=_aH>=0;_n=_aS(1u-u32(_bU),u32(_bU));_aD[_e.x]=_n;if _bU{let _T=_bw[_aH];_b=vec4(f32(_T.x0),f32(_T.y0),f32(_T.x1),f32(_T.y1));}else{_b=vec4(-1e9,-1e9,1e9,1e9);}var _fX=0u;for(var i=0u;i<firstTrailingBit(_i)-1u;i+=1u){let _ga=2u*_i-(1u<<(firstTrailingBit(_i)-i));workgroupBarrier();if _e.x<1u<<(firstTrailingBit(_i)-1u-i){let _fZ=_fX+_e.x*2u;_aD[_ga+_e.x]=_dy(_aD[_fZ],_aD[_fZ+1u]);}_fX=_ga;}workgroupBarrier();
_n=_aS();var _X=_hQ(&_n,_e.x);_dS[_e.x]=_X;workgroupBarrier();let _dU=select(_X-1,_dS[_X],_X>=0);var _P:i32;if _X>=0{_P=i32(_ah.x*_i)+_X;}else if _X+i32(_eQ)>=0{_P=i32(_bT[i32(_i)+_X]);}else{_P=-1;}for(var i=0u;i<firstTrailingBit(_i);i+=1u){if i !=0u{_dS[_e.x]=_X;}_dT[_e.x]=_b;workgroupBarrier();if _X>=0{_b=_eq(_dT[_X],_b);_X=_dS[_X];}workgroupBarrier();}if _X+i32(_eQ)>=0{_b=_eq(_di[i32(_i)+_X],_b);}_dT[_e.x]=_b;workgroupBarrier();if !_bU&&_E.x<_l._dv{let _fY=_cz[_P];let _N=_fY._N;let _dx=_fY.ix;let ix=~_aH;_bL[ix]._N=u32(_N);
_bL[ix]._bq=_bL[_dx]._bq;if _dU>=0{_b=_dT[_dU];}else if _dU+i32(_eQ)>=0{_b=_di[i32(_i)+_dU];}else{_b=vec4(-1e9,-1e9,1e9,1e9);}}if _E.x<_l._dv{_hN[_E.x]=_b;}}`
