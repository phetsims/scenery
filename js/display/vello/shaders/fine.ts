/* eslint-disable */
import ptcl from './shared/ptcl.js';
import blend from './shared/blend.js';
import config from './shared/config.js';
import segment from './shared/segment.js';

export default `struct _aI{_w:i32,_Q:u32}${segment}
${config}
@group(0)@binding(0)
var<uniform>_l:_aL;@group(0)@binding(1)
var<storage>_t:array<_aI>;@group(0)@binding(2)
var<storage>_Q:array<_dm>;${blend}
${ptcl}
const _fI=512;@group(0)@binding(3)
var _bd:texture_storage_2d<rgba8unorm,write>;@group(0)@binding(4)
var<storage>_y:array<u32>;@group(0)@binding(5)
var _eC:texture_2d<f32>;@group(0)@binding(6)
var<storage>_k:array<u32>;@group(0)@binding(7)
var _dE:texture_2d<f32>;fn _hr(_C:u32)->_fc{let _a=_y[_C+1u];let _w=i32(_y[_C+2u]);return _fc(_a,_w);}fn _hq(_C:u32)->_fb{let _a=_y[_C+1u];let _cF=bitcast<f32>(_y[_C+2u]);return _fb(_a,_cF);}fn _hp(_C:u32)->_eb{let _cE=_y[_C+1u];return _eb(_cE);}fn _ho(_C:u32)->_gx{let _dc=_y[_C+1u];let _ak=_dc>>2u;let _bx=_dc&0x3u;let _K=_y[_C+2u];let _bW=bitcast<f32>(_k[_K]);let _eZ=bitcast<f32>(_k[_K+1u]);let _dn=bitcast<f32>(_k[_K+2u]);return _gx(_ak,_bx,_bW,_eZ,_dn);}fn _hn(_C:u32)->_gw{let _dc=_y[_C+1u];let _ak=_dc>>2u;let _bx=_dc&0x3u;let _K=_y[_C+2u];let m0=bitcast<f32>(_k[_K]);let m1=bitcast<f32>(_k[_K+1u]);let m2=bitcast<f32>(_k[_K+2u]);let m3=bitcast<f32>(_k[_K+3u]);let _z=vec4(m0,m1,m2,m3);let _cp=vec2(bitcast<f32>(_k[_K+4u]),bitcast<f32>(_k[_K+5u]));let _aV=bitcast<f32>(_k[_K+6u]);let _aK=bitcast<f32>(_k[_K+7u]);let _fH=_k[_K+8u];let _ad=_fH>>3u;let _aR=_fH&0x7u;return _gw(_ak,_bx,_z,_cp,_aV,_aK,_aR,_ad);}fn _hm(_C:u32)->_gv{let _K=_y[_C+1u];let m0=bitcast<f32>(_k[_K]);let m1=bitcast<f32>(_k[_K+1u]);let m2=bitcast<f32>(_k[_K+2u]);let m3=bitcast<f32>(_k[_K+3u]);let _z=vec4(m0,m1,m2,m3);let _cp=vec2(bitcast<f32>(_k[_K+4u]),bitcast<f32>(_k[_K+5u]));let xy=_k[_K+6u];let _fG=_k[_K+7u];
let x=f32(xy>>16u);let y=f32(xy&0xffffu);let _m=f32(_fG>>16u);let _az=f32(_fG&0xffffu);return _gv(_z,_cp,vec2(x,y),vec2(_m,_az));}fn _hl(_C:u32)->_ea{let _H=_y[_C+1u];let _aQ=bitcast<f32>(_y[_C+2u]);return _ea(_H,_aQ);}fn _bx(t:f32,_M:u32)->f32{let _hj=0u;let _hi=1u;let _hh=2u;switch _M{case 0u:{return clamp(t,0.0,1.0);}case 1u:{return fract(t);}default:{return abs(t-2.0*round(0.5*t));}}}const _V=4u;fn _fJ(_a:_aI,xy:vec2<f32>,_cx:bool)->array<f32,_V>{var _ar:array<f32,_V>;let _hg=f32(_a._w);for(var i=0u;i<_V;i+=1u){_ar[i]=_hg;}var _ci=_a._Q;while _ci !=0u{let _s=_Q[_ci];let y=_s._co.y-xy.y;let y0=clamp(y,0.0,1.0);let y1=clamp(y+_s._aq.y,0.0,1.0);let dy=y0-y1;if dy !=0.0{let _fF=1.0/_s._aq.y;let t0=(y0-y)*_fF;let t1=(y1-y)*_fF;let _fE=_s._co.x-xy.x;let x0=_fE+t0*_s._aq.x;let x1=_fE+t1*_s._aq.x;let _hf=min(x0,x1);let _he=max(x0,x1);for(var i=0u;i<_V;i+=1u){let _fD=f32(i);let _bJ=min(_hf-_fD,1.0)-1.0e-6;let _dI=_he-_fD;let b=min(_dI,1.0);let c=max(b,0.0);let d=max(_bJ,0.0);let a=(b+0.5*(d*d-c*c)-_bJ)/(_dI-_bJ);_ar[i]+=a*dy;}}let _aU=sign(_s._aq.x)*clamp(xy.y-_s._aU+1.0,0.0,1.0);for(var i=0u;i<_V;i+=1u){_ar[i]+=_aU;}_ci=_s._aT;}if _cx{for(var i=0u;i<_V;i+=1u){let a=_ar[i];_ar[i]=abs(a-2.0*round(0.5*a));}}else{for(var i=0u;i<_V;i+=1u){_ar[i]=min(abs(_ar[i]),1.0);}}return _ar;}fn _hk(seg:u32,_cF:f32,xy:vec2<f32>)->array<f32,_V>{var df:array<f32,_V>;for(var i=0u;i<_V;i+=1u){df[i]=1e9;}var _ci=seg;while _ci !=0u{let _s=_Q[_ci];let _aq=_s._aq;let _fC=xy+vec2(0.5,0.5)-_s._co;let _G=1.0/dot(_aq,_aq);for(var i=0u;i<_V;i+=1u){let _db=vec2(_fC.x+f32(i),_fC.y);let t=clamp(dot(_db,_aq)*_G,0.0,1.0);
df[i]=min(df[i],length(_aq*t-_db));}_ci=_s._aT;}for(var i=0u;i<_V;i+=1u){df[i]=clamp(_cF+0.5-df[i],0.0,1.0);}return df;}@compute @workgroup_size(4,16)
fn main(
@builtin(global_invocation_id)_E:vec3<u32>,@builtin(local_invocation_id)_e:vec3<u32>,@builtin(workgroup_id)_ah:vec3<u32>,){let _ap=_ah.y*_l._aB+_ah.x;let xy=vec2(f32(_E.x*_V),f32(_E.y));var rgba:array<vec4<f32>,_V>;for(var i=0u;i<_V;i+=1u){rgba[i]=unpack4x8unorm(_l._ik).wzyx;}var _fx:array<array<u32,_V>,_ei>;var _aY=0u;var _ar:array<f32,_V>;var _C=_ap*_dZ;let _fT=_y[_C];_C+=1u;
while true{let _f=_y[_C];if _f==_do{break;}switch _f{case 1u:{let _aZ=_hr(_C);let _Q=_aZ._a>>1u;let _cx=(_aZ._a&1u)!=0u;let _a=_aI(_aZ._w,_Q);_ar=_fJ(_a,xy,_cx);_C+=3u;}case 2u:{let _al=_hq(_C);_ar=_hk(_al._a,_al._cF,xy);_C+=3u;}case 3u:{for(var i=0u;i<_V;i+=1u){_ar[i]=1.0;}_C+=1u;}case 5u:{let _O=_hp(_C);let fg=unpack4x8unorm(_O._cE).wzyx;for(var i=0u;i<_V;i+=1u){let _bl=fg*_ar[i];rgba[i]=rgba[i]*(1.0-_bl.a)+_bl;}_C+=2u;}case 6u:{let _h=_ho(_C);let d=_h._bW*xy.x+_h._eZ*xy.y+_h._dn;for(var i=0u;i<_V;i+=1u){let _hd=d+_h._bW*f32(i);let x=i32(round(_bx(_hd,_h._bx)*f32(_fI-1)));let _da=textureLoad(_eC,vec2(x,i32(_h._ak)),0);let _bl=_da*_ar[i];rgba[i]=rgba[i]*(1.0-_bl.a)+_bl;}_C+=3u;}case 7u:{let _v=_hn(_C);let _aV=_v._aV;let _aK=_v._aK;let _hc=_v._aR==_gK;let _hb=_v._aR==_gL;let _ha=_v._aR==_gJ;let _fB=(_v._ad&_gI)!=0u;let _fA=select(1.0/_aK,0.0,_hb);let _gZ=select(1.0,-1.0,_fB||(1.0-_aV)<0.0);let _gY=sign(1.0-_aV);for(var i=0u;i<_V;i+=1u){let _cZ=vec2(xy.x+f32(i),xy.y);let _fz=_v._z.xy*_cZ.x+_v._z.zw*_cZ.y+_v._cp;let x=_fz.x;let y=_fz.y;let xx=x*x;let yy=y*y;var t=0.0;var _dF=true;if _hc{let a=_aK-yy;t=sqrt(a)+x;_dF=a>=0.0;}else if _ha{t=(xx+yy)/x;_dF=t>=0.0&&x !=0.0;}else if _aK>1.0{t=sqrt(xx+yy)-x*_fA;}else{let a=xx-yy;t=_gZ*sqrt(a)-x*_fA;_dF=a>=0.0&&t>=0.0;}if _dF{t=_bx(_aV+_gY*t,_v._bx);t=select(t,1.0-t,_fB);let x=i32(round(t*f32(_fI-1)));let _da=textureLoad(_eC,vec2(x,i32(_v._ak)),0);let _bl=_da*_ar[i];rgba[i]=rgba[i]*(1.0-_bl.a)+_bl;}}_C+=3u;}case 8u:{let _aC=_hm(_C);let _fy=_aC._eY+_aC._dY;for(var i=0u;i<_V;i+=1u){let _cZ=vec2(xy.x+f32(i),xy.y);let _dH=_aC._z.xy*_cZ.x+_aC._z.zw*_cZ.y+_aC._cp+_aC._eY;
if all(_dH<_fy)&&_ar[i] !=0.0{let _dG=vec4(max(floor(_dH),_aC._eY),min(ceil(_dH),_fy));let _eE=fract(_dH);let a=_dJ(textureLoad(_dE,vec2<i32>(_dG.xy),0));let b=_dJ(textureLoad(_dE,vec2<i32>(_dG.xw),0));let c=_dJ(textureLoad(_dE,vec2<i32>(_dG.zy),0));let d=_dJ(textureLoad(_dE,vec2<i32>(_dG.zw),0));let _da=mix(mix(a,b,_eE.y),mix(c,d,_eE.y),_eE.x);let _bl=_da*_ar[i];rgba[i]=rgba[i]*(1.0-_bl.a)+_bl;}}_C+=2u;}case 9u:{if _aY<_ei{for(var i=0u;i<_V;i+=1u){_fx[_aY][i]=pack4x8unorm(rgba[i]);rgba[i]=vec4(0.0);}}else{}_aY+=1u;_C+=1u;}case 10u:{let _bI=_hl(_C);_aY-=1u;for(var i=0u;i<_V;i+=1u){var _fw:u32;if _aY<_ei{_fw=_fx[_aY][i];}else{}let bg=unpack4x8unorm(_fw);let fg=rgba[i]*_ar[i]*_bI._aQ;rgba[i]=_iR(bg,fg,_bI._H);}_C+=3u;}case 11u:{_C=_y[_C+1u];}default:{}}}let _eD=vec2<u32>(xy);for(var i=0u;i<_V;i+=1u){let _cj=_eD+vec2(i,0u);if _cj.x<_l._gH&&_cj.y<_l._gG{textureStore(_bd,vec2<i32>(_cj),rgba[i]);}}}fn _dJ(rgba:vec4<f32>)->vec4<f32>{return vec4(rgba.rgb*rgba.a,rgba.a);}`