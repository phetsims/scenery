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
var<storage>_bO:array<_ac>;@group(0)@binding(3)
var<storage>_cu:array<_ff>;@group(0)@binding(4)
var<storage>_J:array<_aJ>;struct _eB{_w:atomic<i32>,_Q:atomic<u32>}@group(0)@binding(5)
var<storage,read_write>_af:_el;@group(0)@binding(6)
var<storage,read_write>_t:array<_eB>;@group(0)@binding(7)
var<storage,read_write>_Q:array<_dm>;struct _ch{_r:f32,a0:f32,a2:f32}const D=0.67;fn _cg(x:f32)->f32{return x*inverseSqrt(sqrt(1.0-D+(D*D*D*D+0.25*x*x)));}const B=0.39;fn _cf(x:f32)->f32{return x*sqrt(1.0-B+(B*B+0.5*x*x));}fn _eA(p0:vec2<f32>,p1:vec2<f32>,p2:vec2<f32>,_cM:f32)->_ch{let _cY=p1-p0;let _cX=p2-p1;let dd=_cY-_cX;let _aW=(p2.x-p0.x)*dd.y-(p2.y-p0.y)*dd.x;let _cW=select(1.0/_aW,1.0e9,abs(_aW)<1.0e-9);let x0=dot(_cY,dd)*_cW;let x2=dot(_cX,dd)*_cW;let _G=abs(_aW/(length(dd)*(x2-x0)));let a0=_cg(x0);let a2=_cg(x2);var _r=0.0;if _G<1e9{let da=abs(a2-a0);let _cV=sqrt(_G);if sign(x0)==sign(x2){_r=_cV;}else{let _bJ=_cM/_cV;_r=_cM/_cg(_bJ);}_r*=da;}return _ch(_r,a0,a2);}fn _ez(p0:vec2<f32>,p1:vec2<f32>,p2:vec2<f32>,t:f32)->vec2<f32>{let mt=1.0-t;return p0*(mt*mt)+(p1*(mt*2.0)+p2*t)*t;}fn _bF(p0:vec2<f32>,p1:vec2<f32>,p2:vec2<f32>,p3:vec2<f32>,t:f32)->vec2<f32>{let mt=1.0-t;return p0*(mt*mt*mt)+(p1*(mt*mt*3.0)+(p2*(mt*3.0)+p3*t)*t)*t;}fn _ey()->u32{var _d=atomicAdd(&_af._Q,1u)+1u;if _d+1u>_l._if{_d=0u;atomicOr(&_af._ab,_gM);}return _d;}const _cU=16u;@compute @workgroup_size(256)
fn main(
@builtin(global_invocation_id)_E:vec3<u32>,){if(atomicLoad(&_af._ab)&(_ek|_fi))!=0u{return;}let ix=_E.x;let _D=_q[_l._du+(ix>>2u)];let _bQ=(ix&3u)*8u;var _bP=(_D>>_bQ)&0xffu;if(_bP&_fd)!=0u{let _R=_cu[_E.x];let _c=_J[_R._N];let _fv=(_R._ad&_id)!=0u;let _b=vec4<i32>(_c._b);let p0=_R.p0;let p1=_R.p1;let p2=_R.p2;let p3=_R.p3;let _cT=3.0*(p2-p1)+p0-p3;let _bE=dot(_cT,_cT);let _av=0.25;let _ce=_av*0.1;let _cS=(_av-_ce);let _ex=432.0*_ce*_ce;var _bi=max(u32(ceil(pow(_bE*(1.0/_ex),1.0/6.0))),1u);_bi=min(_bi,_cU);var _cL:array<_ch,_cU>;var _r=0.0;var _aN=p0;let _aP=1.0/f32(_bi);for(var i=0u;i<_bi;i+=1u){let t=f32(i+1u)*_aP;let _aO=_bF(p0,p1,p2,p3,t);var _aM=_bF(p0,p1,p2,p3,t-0.5*_aP);_aM=2.0*_aM-0.5*(_aN+_aO);let _U=_eA(_aN,_aM,_aO,sqrt(_cS));_cL[i]=_U;_r+=_U._r;_aN=_aO;}let n=max(u32(ceil(_r*(0.5/sqrt(_cS)))),1u);var _S=p0;_aN=p0;let _cR=_r/f32(n);var _bC=1u;var _cc=0.0;for(var i=0u;i<_bi;i+=1u){let t=f32(i+1u)*_aP;let _aO=_bF(p0,p1,p2,p3,t);var _aM=_bF(p0,p1,p2,p3,t-0.5*_aP);_aM=2.0*_aM-0.5*(_aN+_aO);let _U=_cL[i];let u0=_cf(_U.a0);let u2=_cf(_U.a2);let _ew=1.0/(u2-u0);var _cb=f32(_bC)*_cR;while _bC==n||_cb<_cc+_U._r{var _an:vec2<f32>;if _bC==n{_an=p3;}else{let u=(_cb-_cc)/_U._r;let a=mix(_U.a0,_U.a2,u);let au=_cf(a);let t=(au-u0)*_ew;_an=_ez(_aN,_aM,_aO,t);}let _bD=min(_S,_an)-_R._al;let _cQ=max(_S,_an)+_R._al;let dp=_an-_S;let _ev=1.0/dp.x;let _cd=select(dp.x/dp.y,1.0e9,abs(dp.y)<1.0e-9);let SX=1.0/f32(_ct);let SY=1.0/f32(_bg);let c=(_R._al.x+abs(_cd)*(0.5*f32(_bg)+_R._al.y))*SX;let b=_cd;let a=(_S.x-(_S.y-0.5*f32(_bg))*b)*SX;var x0=i32(floor(_bD.x*SX));var x1=i32(floor(_cQ.x*SX)+1.0);var y0=i32(floor(_bD.y*SY));var y1=i32(floor(_cQ.y*SY)+1.0);x0=clamp(x0,_b.x,_b.z);x1=clamp(x1,_b.x,_b.z);y0=clamp(y0,_b.y,_b.w);y1=clamp(y1,_b.y,_b.w);var xc=a+b*f32(y0);let _bK=_b.z-_b.x;var _p=i32(_c._t)+(y0-_b.y)*_bK-_b.x;var _A=i32(floor(_S.x*SX));var _ca=i32(floor(_an.x*SX));if dp.y<0.0{let _bk=_A;_A=_ca;_ca=_bk;}for(var y=y0;y<y1;y+=1){let _eu=f32(y)*f32(_bg);let _cP=max(_A+1,_b.x);if !_fv&&_bD.y<_eu&&_cP<_b.z{let _w=select(-1,1,dp.y<0.0);let _ap=_p+_cP;atomicAdd(&_t[_ap]._w,_w);}var _bB=_ca;if y+1<y1{let _et=f32(y+1)*f32(_bg);let _es=_S.x+(_et-_S.y)*_cd;_bB=i32(floor(_es*SX));}let _cO=min(_A,_bB);let _cN=max(_A,_bB);var _bZ=min(i32(floor(xc-c)),_cO);var _bY=max(i32(ceil(xc+c)),_cN+1);_bZ=clamp(_bZ,x0,x1);_bY=clamp(_bY,x0,x1);var _am:_dm;for(var x=_bZ;x<_bY;x+=1){let _bt=f32(x)*f32(_ct);let _ap=_p+x;
let _bj=_ey();let _dC=atomicExchange(&_t[_ap]._Q,_bj);_am._co=_S;_am._aq=dp;var _aU=0.0;if !_fv{_aU=mix(_S.y,_an.y,(_bt-_S.x)*_ev);if _bD.x<_bt{let p=vec2(_bt,_aU);if dp.x<0.0{_am._aq=p-_S;}else{_am._co=p;_am._aq=_an-p;}if _am._aq.x==0.0{_am._aq.x=sign(dp.x)*1e-9;}}if x<=_cO||_cN<x{_aU=1e9;}}_am._aU=_aU;_am._aT=_dC;_Q[_bj]=_am;}xc+=b;_p+=_bK;_A=_bB;}_bC+=1u;_cb+=_cR;_S=_an;}_cc+=_U._r;_aN=_aO;}}}`
