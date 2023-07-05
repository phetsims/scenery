/* eslint-disable */
import clip from './shared/clip.js';
import bbox from './shared/bbox.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${bbox}
${clip}
@group(0)@binding(0)var<uniform>j:aM;@group(0)@binding(1)var<storage>cT:array<eq>;@group(0)@binding(2)var<storage>bL:array<c6>;@group(0)@binding(3)var<storage,read_write>aK:array<aX>;@group(0)@binding(4)var<storage,read_write>h4:array<fv>;const k=256u;var<workgroup>aJ:array<aX,k>;var<workgroup>bk:array<d,k>;var<workgroup>gk:array<d,k>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)J:I,@builtin(local_invocation_id)f:I,@builtin(workgroup_id)al:I){let cU=cT[J.x].W;let cd=cU>=0;var ah=aX(1u-d(cd),d(cd));aJ[f.x]=ah;for(var e=0u;e<firstTrailingBit(k);e+=1u){workgroupBarrier();if f.x+(1u<<e)<k{let ao=aJ[f.x+(1u<<e)];ah=dQ(ah,ao);}workgroupBarrier();aJ[f.x]=ah;}if f.x==0u{aK[al.x]=ah;}workgroupBarrier();let size=aJ[0].b;ah=aX();if cd&&ah.a==0u{let gl=size-ah.b- 1u;bk[gl]=f.x;gk[gl]=d(cU);}workgroupBarrier();if f.x<size{let W=gk[f.x];let aP=bL[W];let dP=bk[f.x]+al.x*k;let h=vec4(c(aP.F),c(aP.N),c(aP.O),c(aP.R));h4[J.x]=fv(dP,h);}}`
