/* eslint-disable */

export default `struct az{C:vec4f,a0:vec2f}fn cP(P:az,bF:vec2f)->vec2f{return P.C.xy*bF.x+P.C.zw*bF.y+P.a0;}fn e9(P:az)->az{let ia=1./(P.C.x*P.C.w-P.C.y*P.C.z);let e8=ia*vec4(P.C.w,-P.C.y,-P.C.z,P.C.x);let h9=mat2x2(e8.xy,e8.zw)*-P.a0;return az(e8,h9);}fn dw(a:az,b:az)->az{return az(a.C.xyxy*b.C.xxzz+a.C.zwzw*b.C.yyww,a.C.xy*b.a0.x+a.C.zw*b.a0.y+a.a0);}`
