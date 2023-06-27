/* eslint-disable */

export default `struct PathBbox{x0:i32,y0:i32,x1:i32,y1:i32,linewidth:f32,trans_ix:u32,}fn bbox_intersect(a:vec4<f32>,b:vec4<f32>)->vec4<f32>{return vec4(max(a.xy,b.xy),min(a.zw,b.zw));}`
