/* eslint-disable */

export default `struct aP{P:L,b2:B}fn c0(Z:aP,bO:B)->B{return Z.P.xy*bO.x+Z.P.zw*bO.y+Z.b2;}fn fm(Z:aP)->aP{let iv=d/(Z.P.x*Z.P.w-Z.P.y*Z.P.z);let fl=iv*vec4(Z.P.w,-Z.P.y,-Z.P.z,Z.P.x);let iu=mat2x2(fl.xy,fl.zw)*-Z.b2;return aP(fl,iu);}fn dH(a:aP,b:aP)->aP{return aP(a.P.xyxy*b.P.xxzz+a.P.zwzw*b.P.yyww,a.P.xy*b.b2.x+a.P.zw*b.b2.y+a.b2);}fn gJ(ez:j,p:j)->aP{let aO=ez+p*6u;let P=bitcast<L>(vec4(u[aO],u[aO+f],u[aO+2u],u[aO+3u]));let b2=bitcast<B>(vec2(u[aO+4u],u[aO+5u]));return aP(P,b2);}`
