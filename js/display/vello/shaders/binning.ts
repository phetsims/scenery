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
var<storage>_bM:array<_aE>;@group(0)@binding(2)
var<storage>_hV:array<_bY>;@group(0)@binding(3)
var<storage>_hU:array<vec4<f32>>;@group(0)@binding(4)
var<storage,read_write>_hT:array<vec4<f32>>;@group(0)@binding(5)
var<storage,read_write>_af:_em;@group(0)@binding(6)
var<storage,read_write>_bv:array<u32>;struct _eV{_aA:u32,_bn:u32}@group(0)@binding(7)
var<storage,read_write>_co:array<_eV>;const SX=0.00390625;const SY=0.00390625;const _i=256u;const _bW=8u;const _gi=4u;var<workgroup>_bm:array<array<atomic<u32>,_F>,_bW>;var<workgroup>_gd:array<array<u32,_F>,_gi>;var<workgroup>_gc:array<u32,_F>;@compute @workgroup_size(256)
fn main(
@builtin(global_invocation_id)_E:vec3<u32>,@builtin(local_invocation_id)_e:vec3<u32>,@builtin(workgroup_id)_ah:vec3<u32>,){for(var i=0u;i<_bW;i+=1u){atomicStore(&_bm[i][_e.x],0u);}workgroupBarrier();
let _dX=_E.x;var x0=0;var y0=0;var x1=0;var y1=0;if _dX<_l._dx{let _ai=_bM[_dX];var _cB=vec4(-1e9,-1e9,1e9,1e9);if _ai._cr>0u{_cB=_hU[_ai._cr-1u];}let _T=_hV[_ai._N];let pb=vec4<f32>(vec4(_T.x0,_T.y0,_T.x1,_T.y1));let _eU=_er(_cB,pb);
let _b=vec4(_eU.xy,max(_eU.xy,_eU.zw));_hT[_dX]=_b;x0=i32(floor(_b.x*SX));y0=i32(floor(_b.y*SY));x1=i32(ceil(_b.z*SX));y1=i32(ceil(_b.w*SY));}let _cC=i32((_l._aB+_bb-1u)/_bb);let _gh=i32((_l._cs+_ct-1u)/_ct);x0=clamp(x0,0,_cC);y0=clamp(y0,0,_gh);x1=clamp(x1,0,_cC);y1=clamp(y1,0,_gh);if x0==x1{y1=y0;}var x=x0;var y=y0;let _dW=_e.x/32u;let _eT=1u<<(_e.x&31u);while y<y1{atomicOr(&_bm[_dW][y*_cC+x],_eT);x+=1;if x==x1{x=x0;y+=1;}}workgroupBarrier();
var _aA=0u;for(var i=0u;i<_gi;i+=1u){_aA+=countOneBits(atomicLoad(&_bm[i*2u][_e.x]));let _hY=_aA;_aA+=countOneBits(atomicLoad(&_bm[i*2u+1u][_e.x]));let _hX=_aA;let _hW=_hY|(_hX<<16u);_gd[i][_e.x]=_hW;}var _bn=atomicAdd(&_af._dA,_aA);if _bn+_aA>_l._ij{_bn=0u;atomicOr(&_af._ab,_el);}_gc[_e.x]=_bn;_co[_E.x]._aA=_aA;_co[_E.x]._bn=_bn;workgroupBarrier();
x=x0;y=y0;while y<y1{let _dl=y*_cC+x;let _gg=atomicLoad(&_bm[_dW][_dl]);
if(_gg&_eT)!=0u{var _ge=countOneBits(_gg&(_eT-1u));if _dW>0u{let _gf=_dW-1u;let _eS=_gd[_gf/2u][_dl];_ge+=(_eS>>(16u*(_gf&1u)))&0xffffu;}let _d=_l._gG+_gc[_dl];_bv[_d+_ge]=_dX;}x+=1;if x==x1{x=x0;y+=1;}}}`
