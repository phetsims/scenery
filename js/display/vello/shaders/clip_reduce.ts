/* eslint-disable */
import clip from './shared/clip.js';
import bbox from './shared/bbox.js';
import config from './shared/config.js';

export default `${config}
${bbox}
${clip}
@group(0)@binding(0)var<uniform>_m:_aG;@group(0)@binding(1)var<storage>_cI:array<_es>;@group(0)@binding(2)var<storage>_bD:array<_cf>;@group(0)@binding(3)var<storage,read_write>_aq:array<_aS>;@group(0)@binding(4)var<storage,read_write>_hU:array<_fo>;const _j=256u;var<workgroup>_aE:array<_aS,_j>;var<workgroup>_bh:array<u32,_j>;var<workgroup>_gc:array<u32,_j>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)_G:vec3<u32>,@builtin(local_invocation_id)_f:vec3<u32>,@builtin(workgroup_id)_aj:vec3<u32>,){let _aJ=_cI[_G.x]._P;let _cb=_aJ>=0;var _s=_aS(1u-u32(_cb),u32(_cb));_aE[_f.x]=_s;for(var i=0u;i<firstTrailingBit(_j);i+=1u){workgroupBarrier();if _f.x+(1u<<i)<_j{let _Z=_aE[_f.x+(1u<<i)];_s=_dG(_s,_Z);}workgroupBarrier();_aE[_f.x]=_s;}if _f.x==0u{_aq[_aj.x]=_s;}workgroupBarrier();let _F=_aE[0].b;_s=_aS();if _cb&&_s.a==0u{let _gd=_F-_s.b- 1u;_bh[_gd]=_f.x;_gc[_gd]=u32(_aJ);}workgroupBarrier();if _f.x<_F{let _P=_gc[_f.x];let _U=_bD[_P];let _dF=_bh[_f.x]+_aj.x*_j;let _b=vec4(f32(_U.x0),f32(_U.y0),f32(_U.x1),f32(_U.y1));_hU[_G.x]=_fo(_dF,_b);}}`
