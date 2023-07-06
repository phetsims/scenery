/* eslint-disable */
import pathdata_util from './shared/pathdata_util.js';
import transform from './shared/transform.js';
import cubic from './shared/cubic.js';
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
${cubic}
${transform}
@group(0)@binding(0)var<uniform>n:aV;@group(0)@binding(1)var<storage>u:a0;@group(0)@binding(2)var<storage>b7:array<al>;struct hy{J:db,S:db,T:db,V:db,ad:h,bL:j}@group(0)@binding(3)var<storage,read_write>bO:array<hy>;@group(0)@binding(4)var<storage,read_write>eK:array<fz>;var<private>bT:j;${pathdata_util}
fn fQ(x:h)->m{return m(floor(x));}fn fP(x:h)->m{return m(ceil(x));}@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let p=N.x;let K=u[n.dR+(p>>2u)];bT=n.bT;let b9=(p&3u)*8u;var U=ev(K&((f<<b9)-f));U=bS(b7[p>>2u],U);var b8=(K>>b9)&255u;let bM=&bO[U.ac];let ad=bitcast<h>(u[n.iE+U.dN]);if(b8&g7)!=e{(*bM).ad=ad;(*bM).bL=U.bL;}let aW=b8&fw;if aW!=e{var s:B;var v:B;var H:B;var R:B;if(b8&g8)!=e{s=cl(U.aI);v=cl(U.aI+2u);if aW>=c6{H=cl(U.aI+4u);if aW==dP{R=cl(U.aI+6u);}}}else{s=ck(U.aI);v=ck(U.aI+f);if aW>=c6{H=ck(U.aI+2u);if aW==dP{R=ck(U.aI+3u);}}}let Z=gP(n.eB,U.bL);s=c2(Z,s);v=c2(Z,v);var l=vec4(min(s,v),max(s,v));if aW==g9{R=v;H=mix(R,s,d/3.);v=mix(s,R,d/3.);}else if aW>=c6{H=c2(Z,H);l=vec4(min(l.xy,H),max(l.zw,H));if aW==dP{R=c2(Z,R);l=vec4(min(l.xy,R),max(l.zw,R));}else{R=H;H=mix(v,H,d/3.);v=mix(v,s,d/3.);}}var a6=vec2(c,0.);if ad>=c{a6=.5*ad*vec2(length(Z.P.xz),length(Z.P.yw));l+=vec4(-a6,a6);}let aE=j(ad>=c);eK[N.x]=fz(s,v,H,R,a6,U.ac,aE);if l.z>l.x||l.w>l.y{atomicMin(&(*bM).J,fQ(l.x));atomicMin(&(*bM).S,fQ(l.y));atomicMax(&(*bM).T,fP(l.z));atomicMax(&(*bM).V,fP(l.w));}}}`
