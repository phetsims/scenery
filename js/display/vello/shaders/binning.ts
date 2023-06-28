/* eslint-disable */
import bump from './shared/bump.js';
import bbox from './shared/bbox.js';
import drawtag from './shared/drawtag.js';
import config from './shared/config.js';

export default `${config}
${drawtag}
${bbox}
${bump}
@group(0)@binding(0)
var<uniform>_l:_aL;@group(0)@binding(1)
var<storage>_bL:array<_aE>;@group(0)@binding(2)
var<storage>_hT:array<_bX>;@group(0)@binding(3)
var<storage>_hS:array<vec4<f32>>;@group(0)@binding(4)
var<storage,read_write>_hR:array<vec4<f32>>;@group(0)@binding(5)
var<storage,read_write>_af:_el;@group(0)@binding(6)
var<storage,read_write>_bv:array<u32>;struct _eU{_aA:u32,_bn:u32}@group(0)@binding(7)
var<storage,read_write>_cn:array<_eU>;const SX=0.00390625;const SY=0.00390625;const _i=256u;const _bV=8u;const _gh=4u;var<workgroup>_bm:array<array<atomic<u32>,_F>,_bV>;var<workgroup>_gc:array<array<u32,_F>,_gh>;var<workgroup>_gb:array<u32,_F>;@compute @workgroup_size(256)
fn main(
@builtin(global_invocation_id)_E:vec3<u32>,@builtin(local_invocation_id)_e:vec3<u32>,@builtin(workgroup_id)_ah:vec3<u32>,){for(var i=0u;i<_bV;i+=1u){atomicStore(&_bm[i][_e.x],0u);}workgroupBarrier();
let _dW=_E.x;var x0=0;var y0=0;var x1=0;var y1=0;if _dW<_l._dw{let _ai=_bL[_dW];var _cA=vec4(-1e9,-1e9,1e9,1e9);if _ai._cq>0u{_cA=_hS[_ai._cq-1u];}let _T=_hT[_ai._N];let pb=vec4<f32>(vec4(_T.x0,_T.y0,_T.x1,_T.y1));let _eT=_eq(_cA,pb);
let _b=vec4(_eT.xy,max(_eT.xy,_eT.zw));_hR[_dW]=_b;x0=i32(floor(_b.x*SX));y0=i32(floor(_b.y*SY));x1=i32(ceil(_b.z*SX));y1=i32(ceil(_b.w*SY));}let _cB=i32((_l._aB+_bb-1u)/_bb);let _gg=i32((_l._cr+_cs-1u)/_cs);x0=clamp(x0,0,_cB);y0=clamp(y0,0,_gg);x1=clamp(x1,0,_cB);y1=clamp(y1,0,_gg);if x0==x1{y1=y0;}var x=x0;var y=y0;let _dV=_e.x/32u;let _eS=1u<<(_e.x&31u);while y<y1{atomicOr(&_bm[_dV][y*_cB+x],_eS);x+=1;if x==x1{x=x0;y+=1;}}workgroupBarrier();
var _aA=0u;for(var i=0u;i<_gh;i+=1u){_aA+=countOneBits(atomicLoad(&_bm[i*2u][_e.x]));let _hW=_aA;_aA+=countOneBits(atomicLoad(&_bm[i*2u+1u][_e.x]));let _hV=_aA;let _hU=_hW|(_hV<<16u);_gc[i][_e.x]=_hU;}var _bn=atomicAdd(&_af._dz,_aA);if _bn+_aA>_l._ih{_bn=0u;atomicOr(&_af._ab,_ek);}_gb[_e.x]=_bn;_cn[_E.x]._aA=_aA;_cn[_E.x]._bn=_bn;workgroupBarrier();
x=x0;y=y0;while y<y1{let _dk=y*_cB+x;let _gf=atomicLoad(&_bm[_dV][_dk]);
if(_gf&_eS)!=0u{var _gd=countOneBits(_gf&(_eS-1u));if _dV>0u{let _ge=_dV-1u;let _eR=_gc[_ge/2u][_dk];_gd+=(_eR>>(16u*(_ge&1u)))&0xffffu;}let _d=_l._gF+_gb[_dk];_bv[_d+_gd]=_dW;}x+=1;if x==x1{x=x0;y+=1;}}}`
