/* eslint-disable */
import clip from './shared/clip.js';
import bbox from './shared/bbox.js';
import config from './shared/config.js';

export default `${config}
${bbox}
${clip}
@group(0)@binding(0)var<uniform>f:aF;@group(0)@binding(1)var<storage>cK:array<ef>;@group(0)@binding(2)var<storage>bD:array<cY>;@group(0)@binding(3)var<storage,read_write>aD:array<aP>;@group(0)@binding(4)var<storage,read_write>hU:array<fj>;const h=256u;var<workgroup>aC:array<aP,h>;var<workgroup>a9:array<u32,h>;var<workgroup>f6:array<u32,h>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)E:vec3u,@builtin(local_invocation_id)d:vec3u,@builtin(workgroup_id)ae:vec3u,){let cL=cK[E.x].R;let b4=cL>=0;var Y=aP(1u-u32(b4),u32(b4));aC[d.x]=Y;for(var c=0u;c<firstTrailingBit(h);c+=1u){workgroupBarrier();if d.x+(1u<<c)<h{let ai=aC[d.x+(1u<<c)];Y=dF(Y,ai);}workgroupBarrier();aC[d.x]=Y;}if d.x==0u{aD[ae.x]=Y;}workgroupBarrier();let size=aC[0].b;Y=aP();if b4&&Y.a==0u{let f7=size-Y.b- 1u;a9[f7]=d.x;f6[f7]=u32(cL);}workgroupBarrier();if d.x<size{let R=f6[d.x];let aI=bD[R];let dE=a9[d.x]+ae.x*h;let e=vec4(f32(aI.B),f32(aI.I),f32(aI.J),f32(aI.M));hU[E.x]=fj(dE,e);}}`
