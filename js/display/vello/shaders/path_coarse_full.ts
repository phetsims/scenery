/* eslint-disable */
import bump from './shared/bump.js';
import cubic from './shared/cubic.js';
import segment from './shared/segment.js';
import tile from './shared/tile.js';
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';

export default `${config}
${pathtag}
${tile}
${segment}
${cubic}
${bump}
@group(0)@binding(0)
var<uniform>_l:_aL;@group(0)@binding(1)
var<storage>_q:array<u32>;@group(0)@binding(2)
var<storage>_bP:array<_ac>;@group(0)@binding(3)
var<storage>_cv:array<_fg>;@group(0)@binding(4)
var<storage>_J:array<_aJ>;struct _eC{_w:atomic<i32>,_Q:atomic<u32>}@group(0)@binding(5)
var<storage,read_write>_af:_em;@group(0)@binding(6)
var<storage,read_write>_t:array<_eC>;@group(0)@binding(7)
var<storage,read_write>_Q:array<_dn>;struct _ci{_r:f32,a0:f32,a2:f32}const D=0.67;fn _ch(x:f32)->f32{return x*inverseSqrt(sqrt(1.0-D+(D*D*D*D+0.25*x*x)));}const B=0.39;fn _cg(x:f32)->f32{return x*sqrt(1.0-B+(B*B+0.5*x*x));}fn _eB(p0:vec2<f32>,p1:vec2<f32>,p2:vec2<f32>,_cN:f32)->_ci{let _cZ=p1-p0;let _cY=p2-p1;let dd=_cZ-_cY;let _aW=(p2.x-p0.x)*dd.y-(p2.y-p0.y)*dd.x;let _cX=select(1.0/_aW,1.0e9,abs(_aW)<1.0e-9);let x0=dot(_cZ,dd)*_cX;let x2=dot(_cY,dd)*_cX;let _G=abs(_aW/(length(dd)*(x2-x0)));let a0=_ch(x0);let a2=_ch(x2);var _r=0.0;if _G<1e9{let da=abs(a2-a0);let _cW=sqrt(_G);if sign(x0)==sign(x2){_r=_cW;}else{let _bK=_cN/_cW;_r=_cN/_ch(_bK);}_r*=da;}return _ci(_r,a0,a2);}fn _eA(p0:vec2<f32>,p1:vec2<f32>,p2:vec2<f32>,t:f32)->vec2<f32>{let mt=1.0-t;return p0*(mt*mt)+(p1*(mt*2.0)+p2*t)*t;}fn _bF(p0:vec2<f32>,p1:vec2<f32>,p2:vec2<f32>,p3:vec2<f32>,t:f32)->vec2<f32>{let mt=1.0-t;return p0*(mt*mt*mt)+(p1*(mt*mt*3.0)+(p2*(mt*3.0)+p3*t)*t)*t;}fn _ez()->u32{var _d=atomicAdd(&_af._Q,1u)+1u;if _d+1u>_l._ih{_d=0u;atomicOr(&_af._ab,_gN);}return _d;}const _cV=16u;@compute @workgroup_size(256)
fn main(
@builtin(global_invocation_id)_E:vec3<u32>,){if(atomicLoad(&_af._ab)&(_el|_fj))!=0u{return;}let ix=_E.x;let _D=_q[_l._dv+(ix>>2u)];let _bR=(ix&3u)*8u;var _bQ=(_D>>_bR)&0xffu;if(_bQ&_fe)!=0u{let _R=_cv[_E.x];let _c=_J[_R._N];let _fw=(_R._ad&_if)!=0u;let _b=vec4<i32>(_c._b);let p0=_R.p0;let p1=_R.p1;let p2=_R.p2;let p3=_R.p3;let _cU=3.0*(p2-p1)+p0-p3;let _bE=dot(_cU,_cU);let _av=0.25;let _cf=_av*0.1;let _cT=(_av-_cf);let _ey=432.0*_cf*_cf;var _bi=max(u32(ceil(pow(_bE*(1.0/_ey),1.0/6.0))),1u);_bi=min(_bi,_cV);var _cM:array<_ci,_cV>;var _r=0.0;var _aN=p0;let _aP=1.0/f32(_bi);for(var i=0u;i<_bi;i+=1u){let t=f32(i+1u)*_aP;let _aO=_bF(p0,p1,p2,p3,t);var _aM=_bF(p0,p1,p2,p3,t-0.5*_aP);_aM=2.0*_aM-0.5*(_aN+_aO);let _U=_eB(_aN,_aM,_aO,sqrt(_cT));_cM[i]=_U;_r+=_U._r;_aN=_aO;}let n=max(u32(ceil(_r*(0.5/sqrt(_cT)))),1u);var _S=p0;_aN=p0;let _cS=_r/f32(n);var _bC=1u;var _cd=0.0;for(var i=0u;i<_bi;i+=1u){let t=f32(i+1u)*_aP;let _aO=_bF(p0,p1,p2,p3,t);var _aM=_bF(p0,p1,p2,p3,t-0.5*_aP);_aM=2.0*_aM-0.5*(_aN+_aO);let _U=_cM[i];let u0=_cg(_U.a0);let u2=_cg(_U.a2);let _ex=1.0/(u2-u0);var _cc=f32(_bC)*_cS;while _bC==n||_cc<_cd+_U._r{var _an:vec2<f32>;if _bC==n{_an=p3;}else{let u=(_cc-_cd)/_U._r;let a=mix(_U.a0,_U.a2,u);let au=_cg(a);let t=(au-u0)*_ex;_an=_eA(_aN,_aM,_aO,t);}let _bD=min(_S,_an)-_R._al;let _cR=max(_S,_an)+_R._al;let dp=_an-_S;let _ew=1.0/dp.x;let _ce=select(dp.x/dp.y,1.0e9,abs(dp.y)<1.0e-9);let SX=1.0/f32(_cu);let SY=1.0/f32(_bg);let c=(_R._al.x+abs(_ce)*(0.5*f32(_bg)+_R._al.y))*SX;let b=_ce;let a=(_S.x-(_S.y-0.5*f32(_bg))*b)*SX;var x0=i32(floor(_bD.x*SX));var x1=i32(floor(_cR.x*SX)+1.0);var y0=i32(floor(_bD.y*SY));var y1=i32(floor(_cR.y*SY)+1.0);x0=clamp(x0,_b.x,_b.z);x1=clamp(x1,_b.x,_b.z);y0=clamp(y0,_b.y,_b.w);y1=clamp(y1,_b.y,_b.w);var xc=a+b*f32(y0);let _bL=_b.z-_b.x;var _p=i32(_c._t)+(y0-_b.y)*_bL-_b.x;var _A=i32(floor(_S.x*SX));var _cb=i32(floor(_an.x*SX));if dp.y<0.0{let _bk=_A;_A=_cb;_cb=_bk;}for(var y=y0;y<y1;y+=1){let _ev=f32(y)*f32(_bg);let _cQ=max(_A+1,_b.x);if !_fw&&_bD.y<_ev&&_cQ<_b.z{let _w=select(-1,1,dp.y<0.0);let _ap=_p+_cQ;atomicAdd(&_t[_ap]._w,_w);}var _bB=_cb;if y+1<y1{let _eu=f32(y+1)*f32(_bg);let _et=_S.x+(_eu-_S.y)*_ce;_bB=i32(floor(_et*SX));}let _cP=min(_A,_bB);let _cO=max(_A,_bB);var _ca=min(i32(floor(xc-c)),_cP);var _bZ=max(i32(ceil(xc+c)),_cO+1);_ca=clamp(_ca,x0,x1);_bZ=clamp(_bZ,x0,x1);var _am:_dn;for(var x=_ca;x<_bZ;x+=1){let _bt=f32(x)*f32(_cu);let _ap=_p+x;
let _bj=_ez();let _dD=atomicExchange(&_t[_ap]._Q,_bj);_am._cp=_S;_am._aq=dp;var _aU=0.0;if !_fw{_aU=mix(_S.y,_an.y,(_bt-_S.x)*_ew);if _bD.x<_bt{let p=vec2(_bt,_aU);if dp.x<0.0{_am._aq=p-_S;}else{_am._cp=p;_am._aq=_an-p;}if _am._aq.x==0.0{_am._aq.x=sign(dp.x)*1e-9;}}if x<=_cP||_cO<x{_aU=1e9;}}_am._aU=_aU;_am._aT=_dD;_Q[_bj]=_am;}xc+=b;_p+=_bL;_A=_bB;}_bC+=1u;_cc+=_cS;_S=_an;}_cd+=_U._r;_aN=_aO;}}}`
