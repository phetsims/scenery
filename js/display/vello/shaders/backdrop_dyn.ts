/* eslint-disable */
import tile from './shared/tile.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${tile}
@group(0)@binding(0)var<uniform>n:aV;@group(0)@binding(1)var<storage>bP:array<cK>;@group(0)@binding(2)var<storage,read_write>X:array<bv>;const o=256u;var<workgroup>gM:array<j,o>;var<workgroup>c0:array<j,o>;var<workgroup>gL:array<j,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let aH=N.x;var ep=e;if aH<n.cP{let W=bP[aH];gM[k.x]=W.l.z-W.l.x;ep=W.l.w-W.l.y;gL[k.x]=W.X;}c0[k.x]=ep;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x>=(f<<i){ep+=c0[k.x-(f<<i)];}workgroupBarrier();c0[k.x]=ep;}workgroupBarrier();let it=c0[o-f];for(var c1=k.x;c1<it;c1+=o){var av=e;for(var i=e;i<firstTrailingBit(o);i+=f){let bH=av+((o/2u)>>i);if c1>=c0[bH- f]{av=bH;}}let aR=gM[av];if aR>e{var eo=c1-select(e,c0[av- f],av>e);var aK=gL[av]+eo*aR;var a9=X[aK].O;for(var x=f;x<aR;x+=f){aK+=f;a9+=X[aK].O;X[aK].O=a9;}}}}`
