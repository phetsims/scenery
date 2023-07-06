/* eslint-disable */
import tile from './shared/tile.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${tile}
@group(0)@binding(0)var<uniform>n:aV;@group(0)@binding(1)var<storage>bQ:array<cL>;@group(0)@binding(2)var<storage,read_write>Y:array<bw>;const o=256u;var<workgroup>gN:array<j,o>;var<workgroup>c1:array<j,o>;var<workgroup>gM:array<j,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let aH=N.x;var eq=e;if aH<n.cQ{let X=bQ[aH];gN[k.x]=X.l.z-X.l.x;eq=X.l.w-X.l.y;gM[k.x]=X.Y;}c1[k.x]=eq;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x>=(f<<i){eq+=c1[k.x-(f<<i)];}workgroupBarrier();c1[k.x]=eq;}workgroupBarrier();let iu=c1[o-f];for(var c2=k.x;c2<iu;c2+=o){var av=e;for(var i=e;i<firstTrailingBit(o);i+=f){let bI=av+((o/2u)>>i);if c2>=c1[bI- f]{av=bI;}}let aR=gN[av];if aR>e{var ep=c2-select(e,c1[av- f],av>e);var aK=gM[av]+ep*aR;var a9=Y[aK].O;for(var x=f;x<aR;x+=f){aK+=f;a9+=Y[aK].O;Y[aK].O=a9;}}}}`
