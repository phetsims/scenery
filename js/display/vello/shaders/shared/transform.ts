/* eslint-disable */

export default `struct aP{P:L,b1:B}fn c0(Z:aP,bN:B)->B{return Z.P.xy*bN.x+Z.P.zw*bN.y+Z.b1;}fn fk(Z:aP)->aP{let is=d/(Z.P.x*Z.P.w-Z.P.y*Z.P.z);let fj=is*vec4(Z.P.w,-Z.P.y,-Z.P.z,Z.P.x);let ir=mat2x2(fj.xy,fj.zw)*-Z.b1;return aP(fj,ir);}fn dI(a:aP,b:aP)->aP{return aP(a.P.xyxy*b.P.xxzz+a.P.zwzw*b.P.yyww,a.P.xy*b.b1.x+a.P.zw*b.b1.y+a.b1);}fn gJ(ey:j,p:j)->aP{let aO=ey+p*6u;let P=bitcast<L>(vec4(u[aO],u[aO+f],u[aO+2u],u[aO+3u]));let b1=bitcast<B>(vec2(u[aO+4u],u[aO+5u]));return aP(P,b1);}`
