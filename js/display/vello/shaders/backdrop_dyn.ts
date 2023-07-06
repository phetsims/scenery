/* eslint-disable */
import tile from './shared/tile.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${tile}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>bL:array<cJ>;@group(0)@binding(2)var<storage,read_write>X:array<bs>;const o=256u;var<workgroup>gE:array<j,o>;var<workgroup>cZ:array<j,o>;var<workgroup>gD:array<j,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let aG=N.x;var em=e;if aG<n.cN{let W=bL[aG];gE[k.x]=W.l.z-W.l.x;em=W.l.w-W.l.y;gD[k.x]=W.X;}cZ[k.x]=em;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x>=(f<<i){em+=cZ[k.x-(f<<i)];}workgroupBarrier();cZ[k.x]=em;}workgroupBarrier();let ip=cZ[o-f];for(var c_=k.x;c_<ip;c_+=o){var av=e;for(var i=e;i<firstTrailingBit(o);i+=f){let bD=av+((o/2u)>>i);if c_>=cZ[bD- f]{av=bD;}}let ce=gE[av];if ce>e{var el=c_-select(e,cZ[av- f],av>e);var aJ=gD[av]+el*ce;var a6=X[aJ].O;for(var x=f;x<ce;x+=f){aJ+=f;a6+=X[aJ].O;X[aJ].O=a6;}}}}`
