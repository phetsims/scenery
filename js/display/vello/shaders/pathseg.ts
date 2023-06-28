/* eslint-disable */
import transform from './shared/transform.js';
import cubic from './shared/cubic.js';
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';

export default `${config}
${pathtag}
${cubic}
${transform}
@group(0)@binding(0)
var<uniform>_l:_aL;@group(0)@binding(1)
var<storage>_q:array<u32>;@group(0)@binding(2)
var<storage>_bO:array<_ac>;struct _gX{x0:atomic<i32>,y0:atomic<i32>,x1:atomic<i32>,y1:atomic<i32>,_I:f32,_bs:u32}@group(0)@binding(3)
var<storage,read_write>_bw:array<_gX>;@group(0)@binding(4)
var<storage,read_write>_cu:array<_ff>;var<private>_bf:u32;fn _bH(ix:u32)->vec2<f32>{let x=bitcast<f32>(_q[_bf+ix]);let y=bitcast<f32>(_q[_bf+ix+1u]);return vec2(x,y);}fn _bG(ix:u32)->vec2<f32>{let _g=_q[_bf+ix];let x=f32(i32(_g<<16u)>>16u);let y=f32(i32(_g)>>16u);return vec2(x,y);}fn _eK(_cH:u32,ix:u32)->_ay{let _p=_cH+ix*6u;let c0=bitcast<f32>(_q[_p]);let c1=bitcast<f32>(_q[_p+1u]);let c2=bitcast<f32>(_q[_p+2u]);let c3=bitcast<f32>(_q[_p+3u]);let c4=bitcast<f32>(_q[_p+4u]);let c5=bitcast<f32>(_q[_p+5u]);let _z=vec4(c0,c1,c2,c3);let _ba=vec2(c4,c5);return _ay(_z,_ba);}fn _fu(x:f32)->i32{return i32(floor(x));}fn _ft(x:f32)->i32{return i32(ceil(x));}@compute @workgroup_size(256)
fn main(
@builtin(global_invocation_id)_E:vec3<u32>,@builtin(local_invocation_id)_e:vec3<u32>,){let ix=_E.x;let _D=_q[_l._du+(ix>>2u)];_bf=_l._bf;let _bQ=(ix&3u)*8u;var tm=_ec(_D&((1u<<_bQ)-1u));tm=_by(_bO[ix>>2u],tm);var _bP=(_D>>_bQ)&0xffu;let _x=&_bw[tm._N];let _I=bitcast<f32>(_q[_l._ii+tm._dp]);if(_bP&_gA)!=0u{(*_x)._I=_I;(*_x)._bs=tm._bs;}let _aF=_bP&_fd;if _aF !=0u{var p0:vec2<f32>;var p1:vec2<f32>;var p2:vec2<f32>;var p3:vec2<f32>;if(_bP&_gB)!=0u{p0=_bH(tm._au);p1=_bH(tm._au+2u);if _aF>=_cG{p2=_bH(tm._au+4u);if _aF==_dr{p3=_bH(tm._au+6u);}}}else{p0=_bG(tm._au);p1=_bG(tm._au+1u);if _aF>=_cG{p2=_bG(tm._au+2u);if _aF==_dr{p3=_bG(tm._au+3u);}}}let _u=_eK(_l._cH,tm._bs);p0=_cD(_u,p0);p1=_cD(_u,p1);var _b=vec4(min(p0,p1),max(p0,p1));
if _aF==_gC{p3=p1;p2=mix(p3,p0,1.0/3.0);p1=mix(p0,p3,1.0/3.0);}else if _aF>=_cG{p2=_cD(_u,p2);_b=vec4(min(_b.xy,p2),max(_b.zw,p2));if _aF==_dr{p3=_cD(_u,p3);_b=vec4(min(_b.xy,p3),max(_b.zw,p3));}else{p3=p2;p2=mix(p1,p2,1.0/3.0);p1=mix(p1,p0,1.0/3.0);}}var _al=vec2(0.0,0.0);if _I>=0.0{_al=0.5*_I*vec2(length(_u._z.xz),length(_u._z.yw));_b+=vec4(-_al,_al);}let _ad=u32(_I>=0.0);_cu[_E.x]=_ff(p0,p1,p2,p3,_al,tm._N,_ad);

if _b.z>_b.x||_b.w>_b.y{atomicMin(&(*_x).x0,_fu(_b.x));atomicMin(&(*_x).y0,_fu(_b.y));atomicMax(&(*_x).x1,_ft(_b.z));atomicMax(&(*_x).y1,_ft(_b.w));}}}`
