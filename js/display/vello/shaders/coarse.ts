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
var<storage>_bM:array<_aE>;struct _eV{_aA:u32,_bn:u32}@group(0)@binding(3)
var<storage>_hA:array<_eV>;@group(0)@binding(4)
var<storage>_df:array<u32>;@group(0)@binding(5)
var<storage>_J:array<_aJ>;@group(0)@binding(6)
var<storage>_t:array<_aI>;@group(0)@binding(7)
var<storage,read_write>_af:_em;@group(0)@binding(8)
var<storage,read_write>_y:array<u32>;const _i=256u;const _bW=8u;var<workgroup>_bm:array<array<atomic<u32>,_F>,_bW>;var<workgroup>_bu:array<u32,_i>;var<workgroup>_fR:array<u32,_i>;var<workgroup>_de:array<u32,_i>;var<workgroup>_eO:array<u32,_i>;var<workgroup>_fQ:array<u32,_i>;var<workgroup>_fP:array<u32,_i>;var<workgroup>_aX:array<u32,_i>;var<workgroup>_eN:array<u32,_i>;var<private>_L:u32;var<private>_eM:u32;fn _cz(_B:u32){if _L+_B>=_eM{let _hI=_l._aB*_l._cs*_ea;var _dO=_hI+atomicAdd(&_af._y,_fb);if _dO+_fb>_l._ig{_dO=0u;atomicOr(&_af._ab,_io);}_y[_L]=_gl;_y[_L+1u]=_dO;_L=_dO;_eM=_L+(_fb-_gv);}}fn _di(_a:_aI,_I:f32)->bool{_cz(3u);if _I<0.0{let _cy=_I<-1.0;if _a._Q !=0u{let _aZ=_fd(_a._Q,_a._w);_y[_L]=_gu;let _hH=select(_aZ._a<<1u,(_aZ._a<<1u)|1u,_cy);_y[_L+1u]=_hH;_y[_L+2u]=u32(_aZ._w);_L+=3u;}else{if _cy&&(abs(_a._w)&1)==0{return false;}_y[_L]=_gs;_L+=1u;}}else{let _al=_fc(_a._Q,0.5*_I);_y[_L]=_gt;_y[_L+1u]=_al._a;_y[_L+2u]=bitcast<u32>(_al._cG);_L+=3u;}return true;}fn _hM(_O:_ec){_cz(2u);_y[_L]=_gr;_y[_L+1u]=_O._cF;_L+=2u;}fn _fV(ty:u32,_ak:u32,_K:u32){_cz(3u);_y[_L]=ty;_y[_L+1u]=_ak;_y[_L+2u]=_K;_L+=3u;}fn _hL(_K:u32){_cz(2u);_y[_L]=_go;_y[_L+1u]=_K;_L+=2u;}fn _hK(){_cz(1u);_y[_L]=_gn;_L+=1u;}fn _hJ(_bJ:_eb){_cz(3u);_y[_L]=_gm;_y[_L+1u]=_bJ._H;_y[_L+2u]=bitcast<u32>(_bJ._aQ);_L+=3u;}@compute @workgroup_size(256)
fn main(
@builtin(local_invocation_id)_e:vec3<u32>,@builtin(workgroup_id)_ah:vec3<u32>,){let _ab=atomicLoad(&_af._ab);if(_ab&(_el|_fj|_gN))!=0u{return;}let _cC=(_l._aB+_bb-1u)/_bb;let _dl=_cC*_ah.y+_ah.x;let _dS=(_l._dx+_F-1u)/_F;
let _dR=_bb*_ah.x;let _dQ=_ct*_ah.y;let _ax=_e.x % _bb;let _aG=_e.x/_bb;let _hG=(_dQ+_aG)*_l._aB+_dR+_ax;_L=_hG*_ea;_eM=_L+(_ea-_gv);
var _dN=0u;var _aY=0u;var _cx=0u;var _cm=0u;var _cl=0u;var _eQ=0u;var _bT=0u;
var _eP=0u;var _dM=0u;let _fU=_L;_L+=1u;while true{for(var i=0u;i<_bW;i+=1u){atomicStore(&_bm[i][_e.x],0u);}while true{if _bT==_cl&&_cx<_dS{_eQ=_bT;var _j=0u;if _cx+_e.x<_dS{let _cn=(_cx+_e.x)*_F+_dl;let _co=_hA[_cn];_j=_co._aA;_fR[_e.x]=_co._bn;}for(var i=0u;i<firstTrailingBit(_i);i+=1u){_bu[_e.x]=_j;workgroupBarrier();if _e.x>=(1u<<i){_j+=_bu[_e.x-(1u<<i)];}workgroupBarrier();}_bu[_e.x]=_eQ+_j;workgroupBarrier();_bT=_bu[_i-1u];_cx+=_i;}var ix=_cm+_e.x;if ix>=_cl&&ix<_bT{var _dg=0u;for(var i=0u;i<firstTrailingBit(_i);i+=1u){let _bp=_dg+((_F/2u)>>i);if ix>=_bu[_bp-1u]{_dg=_bp;}}ix-=select(_eQ,_bu[_dg-1u],_dg>0u);let _d=_l._gG+_fR[_dg];_de[_e.x]=_df[_d+ix];}_cl=min(_cm+_F,_bT);if _cl-_cm>=_F||(_cl>=_bT&&_cx>=_dS){break;}}var _f=_du;var _aa:u32;if _e.x+_cm<_cl{_aa=_de[_e.x];_f=_q[_l._cJ+_aa];}var _W=0u;
if _f !=_du{let _N=_bM[_aa]._N;let _c=_J[_N];let _bL=_c._b.z-_c._b.x;_eO[_e.x]=_bL;let dx=i32(_c._b.x)-i32(_dR);let dy=i32(_c._b.y)-i32(_dQ);let x0=clamp(dx,0,i32(_bb));let y0=clamp(dy,0,i32(_ct));let x1=clamp(i32(_c._b.z)-i32(_dR),0,i32(_bb));let y1=clamp(i32(_c._b.w)-i32(_dQ),0,i32(_ct));_fQ[_e.x]=u32(x1-x0);_fP[_e.x]=u32(x0)|u32(y0<<16u);_W=u32(x1-x0)*u32(y1-y0);
let _p=_c._t-u32(dy*i32(_bL)+dx);_eN[_e.x]=_p;}_aX[_e.x]=_W;for(var i=0u;i<firstTrailingBit(_F);i+=1u){workgroupBarrier();if _e.x>=(1u<<i){_W+=_aX[_e.x-(1u<<i)];}workgroupBarrier();_aX[_e.x]=_W;}workgroupBarrier();let _dP=_aX[_F-1u];
for(var ix=_e.x;ix<_dP;ix+=_F){var _aj=0u;for(var i=0u;i<firstTrailingBit(_F);i+=1u){let _bp=_aj+((_F/2u)>>i);if ix>=_aX[_bp-1u]{_aj=_bp;}}_aa=_de[_aj];_f=_q[_l._cJ+_aa];let _dY=ix-select(0u,_aX[_aj-1u],_aj>0u);let _m=_fQ[_aj];let _dh=_fP[_aj];let x=(_dh&0xffffu)+_dY % _m;let y=(_dh>>16u)+_dY/_m;let _ap=_eN[_aj]+_eO[_aj]*y+x;let _a=_t[_ap];let _fT=(_f&1u)!=0u;var _fS=false;if _fT{let _hF=(128u<<8u)|3u;let _bq=_bM[_aa]._bq;let dd=_l._fh+_bq;let _H=_q[dd];_fS=_H !=_hF;}let _hE=_a._Q !=0u||(_a._w==0)==_fT||_fS;if _hE{let _hD=_aj/32u;let _hC=1u<<(_aj&31u);atomicOr(&_bm[_hD][y*_bb+x],_hC);}}workgroupBarrier();


var _dL=0u;var _aw=atomicLoad(&_bm[0u][_e.x]);while true{if _aw==0u{_dL+=1u;
if _dL==_bW{break;}_aw=atomicLoad(&_bm[_dL][_e.x]);if _aw==0u{continue;}}let _aj=_dL*32u+firstTrailingBit(_aw);_aa=_de[_aj];
_aw&=_aw-1u;let _at=_q[_l._cJ+_aa];let dm=_bM[_aa];let dd=_l._fh+dm._bq;let di=dm._K;if _dN==0u{let _ap=_eN[_aj]+_eO[_aj]*_aG+_ax;let _a=_t[_ap];switch _at{case 0x44u:{let _I=bitcast<f32>(_df[di]);if _di(_a,_I){let _cF=_q[dd];_hM(_ec(_cF));}}case 0x114u:{let _I=bitcast<f32>(_df[di]);if _di(_a,_I){let _ak=_q[dd];let _K=di+1u;_fV(_gq,_ak,_K);}}case 0x29cu:{let _I=bitcast<f32>(_df[di]);if _di(_a,_I){let _ak=_q[dd];let _K=di+1u;_fV(_gp,_ak,_K);}}case 0x248u:{let _I=bitcast<f32>(_df[di]);if _di(_a,_I){_hL(di+1u);}}case 0x9u:{if _a._Q==0u&&_a._w==0{_dN=_aY+1u;}else{_hK();_eP+=1u;_dM=max(_dM,_eP);}_aY+=1u;}case 0x21u:{_aY-=1u;_di(_a,-1.0);let _H=_q[dd];let _aQ=bitcast<f32>(_q[dd+1u]);_hJ(_eb(_H,_aQ));_eP-=1u;}default:{}}}else{switch _at{case 0x9u:{_aY+=1u;}case 0x21u:{if _aY==_dN{_dN=0u;}_aY-=1u;}default:{}}}}_cm+=_F;if _cm>=_bT&&_cx>=_dS{break;}workgroupBarrier();}if _dR+_ax<_l._aB&&_dQ+_aG<_l._cs{_y[_L]=_dp;if _dM>_ej{let _hB=_dM*_cu*_bg;_y[_fU]=atomicAdd(&_af._H,_hB);}}}`
