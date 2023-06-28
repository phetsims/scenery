/* eslint-disable */
import tile from './shared/tile.js';
import ptcl from './shared/ptcl.js';
import drawtag from './shared/drawtag.js';
import bump from './shared/bump.js';
import config from './shared/config.js';

export default `${config}
${bump}
${drawtag}
${ptcl}
${tile}
@group(0)@binding(0)
var<uniform>_l:_aL;@group(0)@binding(1)
var<storage>_q:array<u32>;@group(0)@binding(2)
var<storage>_bL:array<_aE>;struct _eU{_aA:u32,_bn:u32}@group(0)@binding(3)
var<storage>_hy:array<_eU>;@group(0)@binding(4)
var<storage>_de:array<u32>;@group(0)@binding(5)
var<storage>_J:array<_aJ>;@group(0)@binding(6)
var<storage>_t:array<_aI>;@group(0)@binding(7)
var<storage,read_write>_af:_el;@group(0)@binding(8)
var<storage,read_write>_y:array<u32>;const _i=256u;const _bV=8u;var<workgroup>_bm:array<array<atomic<u32>,_F>,_bV>;var<workgroup>_bu:array<u32,_i>;var<workgroup>_fQ:array<u32,_i>;var<workgroup>_dd:array<u32,_i>;var<workgroup>_eN:array<u32,_i>;var<workgroup>_fP:array<u32,_i>;var<workgroup>_fO:array<u32,_i>;var<workgroup>_aX:array<u32,_i>;var<workgroup>_eM:array<u32,_i>;var<private>_L:u32;var<private>_eL:u32;fn _cy(_B:u32){if _L+_B>=_eL{let _hG=_l._aB*_l._cr*_dZ;var _dN=_hG+atomicAdd(&_af._y,_fa);if _dN+_fa>_l._ie{_dN=0u;atomicOr(&_af._ab,_im);}_y[_L]=_gk;_y[_L+1u]=_dN;_L=_dN;_eL=_L+(_fa-_gu);}}fn _dh(_a:_aI,_I:f32)->bool{_cy(3u);if _I<0.0{let _cx=_I<-1.0;if _a._Q !=0u{let _aZ=_fc(_a._Q,_a._w);_y[_L]=_gt;let _hF=select(_aZ._a<<1u,(_aZ._a<<1u)|1u,_cx);_y[_L+1u]=_hF;_y[_L+2u]=u32(_aZ._w);_L+=3u;}else{if _cx&&(abs(_a._w)&1)==0{return false;}_y[_L]=_gr;_L+=1u;}}else{let _al=_fb(_a._Q,0.5*_I);_y[_L]=_gs;_y[_L+1u]=_al._a;_y[_L+2u]=bitcast<u32>(_al._cF);_L+=3u;}return true;}fn _hK(_O:_eb){_cy(2u);_y[_L]=_gq;_y[_L+1u]=_O._cE;_L+=2u;}fn _fU(ty:u32,_ak:u32,_K:u32){_cy(3u);_y[_L]=ty;_y[_L+1u]=_ak;_y[_L+2u]=_K;_L+=3u;}fn _hJ(_K:u32){_cy(2u);_y[_L]=_gn;_y[_L+1u]=_K;_L+=2u;}fn _hI(){_cy(1u);_y[_L]=_gm;_L+=1u;}fn _hH(_bI:_ea){_cy(3u);_y[_L]=_gl;_y[_L+1u]=_bI._H;_y[_L+2u]=bitcast<u32>(_bI._aQ);_L+=3u;}@compute @workgroup_size(256)
fn main(
@builtin(local_invocation_id)_e:vec3<u32>,@builtin(workgroup_id)_ah:vec3<u32>,){let _ab=atomicLoad(&_af._ab);if(_ab&(_ek|_fi|_gM))!=0u{return;}let _cB=(_l._aB+_bb-1u)/_bb;let _dk=_cB*_ah.y+_ah.x;let _dR=(_l._dw+_F-1u)/_F;
let _dQ=_bb*_ah.x;let _dP=_cs*_ah.y;let _ax=_e.x % _bb;let _aG=_e.x/_bb;let _hE=(_dP+_aG)*_l._aB+_dQ+_ax;_L=_hE*_dZ;_eL=_L+(_dZ-_gu);
var _dM=0u;var _aY=0u;var _cw=0u;var _cl=0u;var _ck=0u;var _eP=0u;var _bS=0u;
var _eO=0u;var _dL=0u;let _fT=_L;_L+=1u;while true{for(var i=0u;i<_bV;i+=1u){atomicStore(&_bm[i][_e.x],0u);}while true{if _bS==_ck&&_cw<_dR{_eP=_bS;var _j=0u;if _cw+_e.x<_dR{let _cm=(_cw+_e.x)*_F+_dk;let _cn=_hy[_cm];_j=_cn._aA;_fQ[_e.x]=_cn._bn;}for(var i=0u;i<firstTrailingBit(_i);i+=1u){_bu[_e.x]=_j;workgroupBarrier();if _e.x>=(1u<<i){_j+=_bu[_e.x-(1u<<i)];}workgroupBarrier();}_bu[_e.x]=_eP+_j;workgroupBarrier();_bS=_bu[_i-1u];_cw+=_i;}var ix=_cl+_e.x;if ix>=_ck&&ix<_bS{var _df=0u;for(var i=0u;i<firstTrailingBit(_i);i+=1u){let _bp=_df+((_F/2u)>>i);if ix>=_bu[_bp-1u]{_df=_bp;}}ix-=select(_eP,_bu[_df-1u],_df>0u);let _d=_l._gF+_fQ[_df];_dd[_e.x]=_de[_d+ix];}_ck=min(_cl+_F,_bS);if _ck-_cl>=_F||(_ck>=_bS&&_cw>=_dR){break;}}var _f=_dt;var _aa:u32;if _e.x+_cl<_ck{_aa=_dd[_e.x];_f=_q[_l._cI+_aa];}var _W=0u;
if _f !=_dt{let _N=_bL[_aa]._N;let _c=_J[_N];let _bK=_c._b.z-_c._b.x;_eN[_e.x]=_bK;let dx=i32(_c._b.x)-i32(_dQ);let dy=i32(_c._b.y)-i32(_dP);let x0=clamp(dx,0,i32(_bb));let y0=clamp(dy,0,i32(_cs));let x1=clamp(i32(_c._b.z)-i32(_dQ),0,i32(_bb));let y1=clamp(i32(_c._b.w)-i32(_dP),0,i32(_cs));_fP[_e.x]=u32(x1-x0);_fO[_e.x]=u32(x0)|u32(y0<<16u);_W=u32(x1-x0)*u32(y1-y0);
let _p=_c._t-u32(dy*i32(_bK)+dx);_eM[_e.x]=_p;}_aX[_e.x]=_W;for(var i=0u;i<firstTrailingBit(_F);i+=1u){workgroupBarrier();if _e.x>=(1u<<i){_W+=_aX[_e.x-(1u<<i)];}workgroupBarrier();_aX[_e.x]=_W;}workgroupBarrier();let _dO=_aX[_F-1u];
for(var ix=_e.x;ix<_dO;ix+=_F){var _aj=0u;for(var i=0u;i<firstTrailingBit(_F);i+=1u){let _bp=_aj+((_F/2u)>>i);if ix>=_aX[_bp-1u]{_aj=_bp;}}_aa=_dd[_aj];_f=_q[_l._cI+_aa];let _dX=ix-select(0u,_aX[_aj-1u],_aj>0u);let _m=_fP[_aj];let _dg=_fO[_aj];let x=(_dg&0xffffu)+_dX % _m;let y=(_dg>>16u)+_dX/_m;let _ap=_eM[_aj]+_eN[_aj]*y+x;let _a=_t[_ap];let _fS=(_f&1u)!=0u;var _fR=false;if _fS{let _hD=(128u<<8u)|3u;let _bq=_bL[_aa]._bq;let dd=_l._fg+_bq;let _H=_q[dd];_fR=_H !=_hD;}let _hC=_a._Q !=0u||(_a._w==0)==_fS||_fR;if _hC{let _hB=_aj/32u;let _hA=1u<<(_aj&31u);atomicOr(&_bm[_hB][y*_bb+x],_hA);}}workgroupBarrier();


var _dK=0u;var _aw=atomicLoad(&_bm[0u][_e.x]);while true{if _aw==0u{_dK+=1u;
if _dK==_bV{break;}_aw=atomicLoad(&_bm[_dK][_e.x]);if _aw==0u{continue;}}let _aj=_dK*32u+firstTrailingBit(_aw);_aa=_dd[_aj];
_aw&=_aw-1u;let _at=_q[_l._cI+_aa];let dm=_bL[_aa];let dd=_l._fg+dm._bq;let di=dm._K;if _dM==0u{let _ap=_eM[_aj]+_eN[_aj]*_aG+_ax;let _a=_t[_ap];switch _at{case 0x44u:{let _I=bitcast<f32>(_de[di]);if _dh(_a,_I){let _cE=_q[dd];_hK(_eb(_cE));}}case 0x114u:{let _I=bitcast<f32>(_de[di]);if _dh(_a,_I){let _ak=_q[dd];let _K=di+1u;_fU(_gp,_ak,_K);}}case 0x29cu:{let _I=bitcast<f32>(_de[di]);if _dh(_a,_I){let _ak=_q[dd];let _K=di+1u;_fU(_go,_ak,_K);}}case 0x248u:{let _I=bitcast<f32>(_de[di]);if _dh(_a,_I){_hJ(di+1u);}}case 0x9u:{if _a._Q==0u&&_a._w==0{_dM=_aY+1u;}else{_hI();_eO+=1u;_dL=max(_dL,_eO);}_aY+=1u;}case 0x21u:{_aY-=1u;_dh(_a,-1.0);let _H=_q[dd];let _aQ=bitcast<f32>(_q[dd+1u]);_hH(_ea(_H,_aQ));_eO-=1u;}default:{}}}else{switch _at{case 0x9u:{_aY+=1u;}case 0x21u:{if _aY==_dM{_dM=0u;}_aY-=1u;}default:{}}}}_cl+=_F;if _cl>=_bS&&_cw>=_dR{break;}workgroupBarrier();}if _dQ+_ax<_l._aB&&_dP+_aG<_l._cr{_y[_L]=_do;if _dL>_ei{let _hz=_dL*_ct*_bg;_y[_fT]=atomicAdd(&_af._H,_hz);}}}`
