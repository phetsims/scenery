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
@group(0)@binding(0)var<uniform>n:aV;@group(0)@binding(1)var<storage>u:a0;@group(0)@binding(2)var<storage>b8:array<al>;struct hz{J:dc,S:dc,T:dc,V:dc,ad:h,bM:j}@group(0)@binding(3)var<storage,read_write>bP:array<hz>;@group(0)@binding(4)var<storage,read_write>eL:array<fA>;${transform}
var<private>bU:j;${pathdata_util}
fn fR(x:h)->m{return m(floor(x));}fn fQ(x:h)->m{return m(ceil(x));}@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let p=N.x;let K=u[n.dS+(p>>2u)];bU=n.bU;let ca=(p&3u)*8u;var U=ew(K&((f<<ca)-f));U=bT(b8[p>>2u],U);var b9=(K>>ca)&255u;let bN=&bP[U.ac];let ad=bitcast<h>(u[n.iF+U.dO]);if(b9&g8)!=e{(*bN).ad=ad;(*bN).bM=U.bM;}let aW=b9&fx;if aW!=e{var s:B;var v:B;var H:B;var R:B;if(b9&g9)!=e{s=cm(U.aI);v=cm(U.aI+2u);if aW>=c7{H=cm(U.aI+4u);if aW==dQ{R=cm(U.aI+6u);}}}else{s=cl(U.aI);v=cl(U.aI+f);if aW>=c7{H=cl(U.aI+2u);if aW==dQ{R=cl(U.aI+3u);}}}let Z=gQ(n.eC,U.bM);s=c3(Z,s);v=c3(Z,v);var l=vec4(min(s,v),max(s,v));if aW==ha{R=v;H=mix(R,s,d/3.);v=mix(s,R,d/3.);}else if aW>=c7{H=c3(Z,H);l=vec4(min(l.xy,H),max(l.zw,H));if aW==dQ{R=c3(Z,R);l=vec4(min(l.xy,R),max(l.zw,R));}else{R=H;H=mix(v,H,d/3.);v=mix(v,s,d/3.);}}var a6=vec2(c,0.);if ad>=c{a6=.5*ad*vec2(length(Z.Q.xz),length(Z.Q.yw));l+=vec4(-a6,a6);}let aF=j(ad>=c);eL[N.x]=fA(s,v,H,R,a6,U.ac,aF);if l.z>l.x||l.w>l.y{atomicMin(&(*bN).J,fR(l.x));atomicMin(&(*bN).S,fR(l.y));atomicMax(&(*bN).T,fQ(l.z));atomicMax(&(*bN).V,fQ(l.w));}}}`
