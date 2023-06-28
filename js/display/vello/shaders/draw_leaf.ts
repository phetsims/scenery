/* eslint-disable */
import transform from './shared/transform.js';
import bbox from './shared/bbox.js';
import drawtag from './shared/drawtag.js';
import clip from './shared/clip.js';
import config from './shared/config.js';

export default `${config}
${clip}
${drawtag}
${bbox}
${transform}
@group(0)@binding(0)
var<uniform>_l:_aL;@group(0)@binding(1)
var<storage>_q:array<u32>;@group(0)@binding(2)
var<storage>_ao:array<_aE>;@group(0)@binding(3)
var<storage>_T:array<_bX>;@group(0)@binding(4)
var<storage,read_write>_ai:array<_aE>;@group(0)@binding(5)
var<storage,read_write>_k:array<u32>;@group(0)@binding(6)
var<storage,read_write>_cz:array<_ej>;const _i=256u;fn _eK(_cH:u32,ix:u32)->_ay{let _p=_cH+ix*6u;let c0=bitcast<f32>(_q[_p]);let c1=bitcast<f32>(_q[_p+1u]);let c2=bitcast<f32>(_q[_p+2u]);let c3=bitcast<f32>(_q[_p+3u]);let c4=bitcast<f32>(_q[_p+4u]);let c5=bitcast<f32>(_q[_p+5u]);let _z=vec4(c0,c1,c2,c3);let _ba=vec2(c4,c5);return _ay(_z,_ba);}var<workgroup>_as:array<_aE,_i>;@compute @workgroup_size(256)
fn main(
@builtin(global_invocation_id)_E:vec3<u32>,@builtin(local_invocation_id)_e:vec3<u32>,@builtin(workgroup_id)_ah:vec3<u32>,){let ix=_E.x;
var agg=_ic();if _e.x<_ah.x{agg=_ao[_e.x];}_as[_e.x]=agg;for(var i=0u;i<firstTrailingBit(_i);i+=1u){workgroupBarrier();if _e.x+(1u<<i)<_i{let _Y=_as[_e.x+(1u<<i)];agg=_eh(agg,_Y);}workgroupBarrier();_as[_e.x]=agg;}workgroupBarrier();var m=_as[0];workgroupBarrier();let _D=_q[_l._cI+ix];agg=_gE(_D);_as[_e.x]=agg;for(var i=0u;i<firstTrailingBit(_i);i+=1u){workgroupBarrier();if _e.x>=1u<<i{let _Y=_as[_e.x-(1u<<i)];agg=_eh(agg,_Y);}workgroupBarrier();_as[_e.x]=agg;}workgroupBarrier();if _e.x>0u{m=_eh(m,_as[_e.x-1u]);}_ai[ix]=m;let dd=_l._fg+m._bq;let di=m._K;if _D==_fe||_D==_eg||
_D==_ef||_D==_ee||
_D==_ds{let _b=_T[m._N];





let _iW=u32(_b._I>=0.0);var _u=_ay();var _I=_b._I;if _I>=0.0||_D==_eg||_D==_ef||
_D==_ee{_u=_eK(_l._cH,_b._bs);}if _I>=0.0{let _z=_u._z;_I*=sqrt(abs(_z.x*_z.w-_z.y*_z.z));}switch _D{case 0x44u:{_k[di]=bitcast<u32>(_I);}case 0x114u:{_k[di]=bitcast<u32>(_I);var p0=bitcast<vec2<f32>>(vec2(_q[dd+1u],_q[dd+2u]));var p1=bitcast<vec2<f32>>(vec2(_q[dd+3u],_q[dd+4u]));p0=_cD(_u,p0);p1=_cD(_u,p1);let _eJ=p1-p0;let _G=1.0/dot(_eJ,_eJ);let _eI=_eJ*_G;let _dn=-dot(p0,_eI);_k[di+1u]=bitcast<u32>(_eI.x);_k[di+2u]=bitcast<u32>(_eI.y);_k[di+3u]=bitcast<u32>(_dn);}case 0x29cu:{let _eH=1.0/f32(1<<12u);_k[di]=bitcast<u32>(_I);var p0=bitcast<vec2<f32>>(vec2(_q[dd+1u],_q[dd+2u]));var p1=bitcast<vec2<f32>>(vec2(_q[dd+3u],_q[dd+4u]));var r0=bitcast<f32>(_q[dd+5u]);var r1=bitcast<f32>(_q[dd+6u]);let _fL=_eX(_u);
var _bR=_ay();var _aV=0.0;var _aK=0.0;var _aR=0u;var _ad=0u;if abs(r0-r1)<=_eH{_aR=_gK;let _cv=r0/distance(p0,p1);_bR=_dl(
_fN(p0,p1),_fL
);_aK=_cv*_cv;}else{_aR=_il;if all(p0==p1){_aR=_gL;
p0+=_eH;}if r1==0.0{_ad|=_gI;let _hx=p0;p0=p1;p1=_hx;let _hw=r0;r0=r1;r1=_hw;}_aV=r0/(r0-r1);let cf=(1.0-_aV)*p0+_aV*p1;_aK=r1/(distance(cf,p1));let _eG=_dl(
_fN(cf,p1),_fL
);var _eF=_eG;
if abs(_aK-1.0)<=_eH{_aR=_gJ;let _G=0.5*abs(1.0-_aV);_eF=_dl(
_ay(vec4(_G,0.0,0.0,_G),vec2(0.0)),_eG
);}else{let a=_aK*_aK-1.0;let _fK=abs(1.0-_aV)/a;let _hv=_aK*_fK;let _hu=sqrt(abs(a))*_fK;_eF=_dl(
_ay(vec4(_hv,0.0,0.0,_hu),vec2(0.0)),_eG
);}_bR=_eF;}_k[di+1u]=bitcast<u32>(_bR._z.x);_k[di+2u]=bitcast<u32>(_bR._z.y);_k[di+3u]=bitcast<u32>(_bR._z.z);_k[di+4u]=bitcast<u32>(_bR._z.w);_k[di+5u]=bitcast<u32>(_bR._ba.x);_k[di+6u]=bitcast<u32>(_bR._ba.y);_k[di+7u]=bitcast<u32>(_aV);_k[di+8u]=bitcast<u32>(_aK);_k[di+9u]=bitcast<u32>((_ad<<3u)|_aR);}case 0x248u:{_k[di]=bitcast<u32>(_I);let _o=_eX(_u);_k[di+1u]=bitcast<u32>(_o._z.x);_k[di+2u]=bitcast<u32>(_o._z.y);_k[di+3u]=bitcast<u32>(_o._z.z);_k[di+4u]=bitcast<u32>(_o._z.w);_k[di+5u]=bitcast<u32>(_o._ba.x);_k[di+6u]=bitcast<u32>(_o._ba.y);_k[di+7u]=_q[dd];_k[di+8u]=_q[dd+1u];}default:{}}}if _D==_ds||_D==_ed{var _N=~ix;if _D==_ds{_N=m._N;}_cz[m._cq]=_ej(ix,i32(_N));}}fn _fN(p0:vec2<f32>,p1:vec2<f32>)->_ay{let _ht=_fM(p0,p1);let _o=_eX(_ht);let _hs=_fM(vec2(0.0),vec2(1.0,0.0));return _dl(_hs,_o);}fn _fM(p0:vec2<f32>,p1:vec2<f32>)->_ay{return _ay(
vec4(p1.y-p0.y,p0.x-p1.x,p1.x-p0.x,p1.y-p0.y),vec2(p0.x,p0.y)
);}`
