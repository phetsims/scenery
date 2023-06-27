/* eslint-disable */
import clip from './shared/clip.js';
import bbox from './shared/bbox.js';
import config from './shared/config.js';

export default `${config}
${bbox}
${clip}
@group(0)@binding(0)
var<uniform>_l:_aL;@group(0)@binding(1)
var<storage>_cA:array<_ek>;@group(0)@binding(2)
var<storage>_bw:array<_bY>;@group(0)@binding(3)
var<storage,read_write>_ao:array<_aS>;@group(0)@binding(4)
var<storage,read_write>_hN:array<_fi>;const _i=256u;var<workgroup>_aD:array<_aS,_i>;var<workgroup>_be:array<u32,_i>;var<workgroup>_fW:array<u32,_i>;@compute @workgroup_size(256)
fn main(
@builtin(global_invocation_id)_E:vec3<u32>,@builtin(local_invocation_id)_e:vec3<u32>,@builtin(workgroup_id)_ah:vec3<u32>,){let _aH=_cA[_E.x]._N;let _bV=_aH>=0;var _n=_aS(1u-u32(_bV),u32(_bV));
_aD[_e.x]=_n;for(var i=0u;i<firstTrailingBit(_i);i+=1u){workgroupBarrier();if _e.x+(1u<<i)<_i{let _Y=_aD[_e.x+(1u<<i)];_n=_dz(_n,_Y);}workgroupBarrier();_aD[_e.x]=_n;}if _e.x==0u{_ao[_ah.x]=_n;}workgroupBarrier();let _B=_aD[0].b;_n=_aS();if _bV&&_n.a==0u{let _fX=_B-_n.b-1u;_be[_fX]=_e.x;_fW[_fX]=u32(_aH);}workgroupBarrier();
if _e.x<_B{let _N=_fW[_e.x];let _T=_bw[_N];let _dy=_be[_e.x]+_ah.x*_i;let _b=vec4(f32(_T.x0),f32(_T.y0),f32(_T.x1),f32(_T.y1));_hN[_E.x]=_fi(_dy,_b);}}`
