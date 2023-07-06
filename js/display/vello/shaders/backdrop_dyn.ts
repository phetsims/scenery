/* eslint-disable */
import tile from './shared/tile.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${tile}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>bM:array<cJ>;@group(0)@binding(2)var<storage,read_write>W:array<bs>;const o=256u;var<workgroup>gF:array<j,o>;var<workgroup>cZ:array<j,o>;var<workgroup>gE:array<j,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let aG=N.x;var en=e;if aG<n.cN{let X=bM[aG];gF[k.x]=X.l.z-X.l.x;en=X.l.w-X.l.y;gE[k.x]=X.W;}cZ[k.x]=en;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x>=(f<<i){en+=cZ[k.x-(f<<i)];}workgroupBarrier();cZ[k.x]=en;}workgroupBarrier();let ip=cZ[o-f];for(var c_=k.x;c_<ip;c_+=o){var av=e;for(var i=e;i<firstTrailingBit(o);i+=f){let bE=av+((o/2u)>>i);if c_>=cZ[bE- f]{av=bE;}}let cg=gF[av];if cg>e{var em=c_-select(e,cZ[av- f],av>e);var aJ=gE[av]+em*cg;var a6=W[aJ].O;for(var x=f;x<cg;x+=f){aJ+=f;a6+=W[aJ].O;W[aJ].O=a6;}}}}`
