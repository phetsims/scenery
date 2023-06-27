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
var<storage>_T:array<_bY>;@group(0)@binding(4)
var<storage,read_write>_ai:array<_aE>;@group(0)@binding(5)
var<storage,read_write>_k:array<u32>;@group(0)@binding(6)
var<storage,read_write>_cA:array<_ek>;const _i=256u;fn _eL(_cI:u32,ix:u32)->_ay{let _p=_cI+ix*6u;let c0=bitcast<f32>(_q[_p]);let c1=bitcast<f32>(_q[_p+1u]);let c2=bitcast<f32>(_q[_p+2u]);let c3=bitcast<f32>(_q[_p+3u]);let c4=bitcast<f32>(_q[_p+4u]);let c5=bitcast<f32>(_q[_p+5u]);let _z=vec4(c0,c1,c2,c3);let _ba=vec2(c4,c5);return _ay(_z,_ba);}var<workgroup>_as:array<_aE,_i>;@compute @workgroup_size(256)
fn main(
@builtin(global_invocation_id)_E:vec3<u32>,@builtin(local_invocation_id)_e:vec3<u32>,@builtin(workgroup_id)_ah:vec3<u32>,){let ix=_E.x;
var agg=_ie();if _e.x<_ah.x{agg=_ao[_e.x];}_as[_e.x]=agg;for(var i=0u;i<firstTrailingBit(_i);i+=1u){workgroupBarrier();if _e.x+(1u<<i)<_i{let _Y=_as[_e.x+(1u<<i)];agg=_ei(agg,_Y);}workgroupBarrier();_as[_e.x]=agg;}workgroupBarrier();var m=_as[0];workgroupBarrier();let _D=_q[_l._cJ+ix];agg=_gF(_D);_as[_e.x]=agg;for(var i=0u;i<firstTrailingBit(_i);i+=1u){workgroupBarrier();if _e.x>=1u<<i{let _Y=_as[_e.x-(1u<<i)];agg=_ei(agg,_Y);}workgroupBarrier();_as[_e.x]=agg;}workgroupBarrier();if _e.x>0u{m=_ei(m,_as[_e.x-1u]);}_ai[ix]=m;let dd=_l._fh+m._bq;let di=m._K;if _D==_ff||_D==_eh||
_D==_eg||_D==_ef||
_D==_dt{let _b=_T[m._N];





let _iY=u32(_b._I>=0.0);var _u=_ay();var _I=_b._I;if _I>=0.0||_D==_eh||_D==_eg||
_D==_ef{_u=_eL(_l._cI,_b._bs);}if _I>=0.0{let _z=_u._z;_I*=sqrt(abs(_z.x*_z.w-_z.y*_z.z));}switch _D{case 0x44u:{_k[di]=bitcast<u32>(_I);}case 0x114u:{_k[di]=bitcast<u32>(_I);var p0=bitcast<vec2<f32>>(vec2(_q[dd+1u],_q[dd+2u]));var p1=bitcast<vec2<f32>>(vec2(_q[dd+3u],_q[dd+4u]));p0=_cE(_u,p0);p1=_cE(_u,p1);let _eK=p1-p0;let _G=1.0/dot(_eK,_eK);let _eJ=_eK*_G;let _do=-dot(p0,_eJ);_k[di+1u]=bitcast<u32>(_eJ.x);_k[di+2u]=bitcast<u32>(_eJ.y);_k[di+3u]=bitcast<u32>(_do);}case 0x29cu:{let _eI=1.0/f32(1<<12u);_k[di]=bitcast<u32>(_I);var p0=bitcast<vec2<f32>>(vec2(_q[dd+1u],_q[dd+2u]));var p1=bitcast<vec2<f32>>(vec2(_q[dd+3u],_q[dd+4u]));var r0=bitcast<f32>(_q[dd+5u]);var r1=bitcast<f32>(_q[dd+6u]);let _fM=_eY(_u);
var _bS=_ay();var _aV=0.0;var _aK=0.0;var _aR=0u;var _ad=0u;if abs(r0-r1)<=_eI{_aR=_gL;let _cw=r0/distance(p0,p1);_bS=_dm(
_fO(p0,p1),_fM
);_aK=_cw*_cw;}else{_aR=_in;if all(p0==p1){_aR=_gM;
p0+=_eI;}if r1==0.0{_ad|=_gJ;let _hz=p0;p0=p1;p1=_hz;let _hy=r0;r0=r1;r1=_hy;}_aV=r0/(r0-r1);let cf=(1.0-_aV)*p0+_aV*p1;_aK=r1/(distance(cf,p1));let _eH=_dm(
_fO(cf,p1),_fM
);var _eG=_eH;
if abs(_aK-1.0)<=_eI{_aR=_gK;let _G=0.5*abs(1.0-_aV);_eG=_dm(
_ay(vec4(_G,0.0,0.0,_G),vec2(0.0)),_eH
);}else{let a=_aK*_aK-1.0;let _fL=abs(1.0-_aV)/a;let _hx=_aK*_fL;let _hw=sqrt(abs(a))*_fL;_eG=_dm(
_ay(vec4(_hx,0.0,0.0,_hw),vec2(0.0)),_eH
);}_bS=_eG;}_k[di+1u]=bitcast<u32>(_bS._z.x);_k[di+2u]=bitcast<u32>(_bS._z.y);_k[di+3u]=bitcast<u32>(_bS._z.z);_k[di+4u]=bitcast<u32>(_bS._z.w);_k[di+5u]=bitcast<u32>(_bS._ba.x);_k[di+6u]=bitcast<u32>(_bS._ba.y);_k[di+7u]=bitcast<u32>(_aV);_k[di+8u]=bitcast<u32>(_aK);_k[di+9u]=bitcast<u32>((_ad<<3u)|_aR);}case 0x248u:{_k[di]=bitcast<u32>(_I);let _o=_eY(_u);_k[di+1u]=bitcast<u32>(_o._z.x);_k[di+2u]=bitcast<u32>(_o._z.y);_k[di+3u]=bitcast<u32>(_o._z.z);_k[di+4u]=bitcast<u32>(_o._z.w);_k[di+5u]=bitcast<u32>(_o._ba.x);_k[di+6u]=bitcast<u32>(_o._ba.y);_k[di+7u]=_q[dd];_k[di+8u]=_q[dd+1u];}default:{}}}if _D==_dt||_D==_ee{var _N=~ix;if _D==_dt{_N=m._N;}_cA[m._cr]=_ek(ix,i32(_N));}}fn _fO(p0:vec2<f32>,p1:vec2<f32>)->_ay{let _hv=_fN(p0,p1);let _o=_eY(_hv);let _hu=_fN(vec2(0.0),vec2(1.0,0.0));return _dm(_hu,_o);}fn _fN(p0:vec2<f32>,p1:vec2<f32>)->_ay{return _ay(
vec4(p1.y-p0.y,p0.x-p1.x,p1.x-p0.x,p1.y-p0.y),vec2(p0.x,p0.y)
);}`
