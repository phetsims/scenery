// Copyright 2023, University of Colorado Boulder

/**
 * Represents an abstract rendering program, that may be location-varying
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { Color, scenery } from '../../imports.js';
import { Shape } from '../../../../kite/js/imports.js';
import Vector2 from '../../../../dot/js/Vector2.js';
import Matrix4 from '../../../../dot/js/Matrix4.js';
import Matrix3 from '../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../dot/js/Vector4.js';
import Utils from '../../../../dot/js/Utils.js';
import Vector3 from '../../../../dot/js/Vector3.js';

export type FillRule = 'nonzero' | 'evenodd';

export enum RenderComposeType {
  Over = 0,
  In = 1,
  Out = 2,
  Atop = 3,
  Xor = 4,
  Plus = 5,
  PlusLighter = 6
}

export enum RenderBlendType {
  Normal = 0,
  Multiply = 1,
  Screen = 2,
  Overlay = 3,
  Darken = 4,
  Lighten = 5,
  ColorDodge = 6,
  ColorBurn = 7,
  HardLight = 8,
  SoftLight = 9,
  Difference = 10,
  Exclusion = 11,
  Hue = 12,
  Saturation = 13,
  Color = 14,
  Luminosity = 15
}

export enum RenderExtend {
  Pad = 0,
  Reflect = 1,
  Repeat = 2
}

const alwaysTrue: ( renderPath: RenderPath ) => boolean = _.constant( true );

export default abstract class RenderProgram {
  public abstract isFullyTransparent(): boolean;

  public abstract isFullyOpaque(): boolean;

  public abstract simplify( pathTest?: ( renderPath: RenderPath ) => boolean ): RenderProgram;

  // Premultiplied linear RGB, ignoring the path
  public abstract evaluate( point: Vector2, pathTest?: ( renderPath: RenderPath ) => boolean ): Vector4;

  public abstract toRecursiveString( indent: string ): string;

  public abstract equals( other: RenderProgram ): boolean;

  public depthFirst( callback: ( program: RenderProgram ) => void ): void {
    callback( this );
  }

  public slowRasterizeToURL( width: number, height: number, n: number ): string {
    return this.slowRasterizeToCanvas( width, height, n ).toDataURL();
  }

  public slowRasterizeToCanvas( width: number, height: number, n: number ): HTMLCanvasElement {
    const imageData = this.slowRasterizeImageData( width, height, n );
    const canvas = document.createElement( 'canvas' );
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext( '2d' )!;
    context.putImageData( imageData, 0, 0 );
    return canvas;
  }

  public slowRasterizeImageData( width: number, height: number, n: number ): ImageData {
    const imageData = new ImageData( width, height, { colorSpace: 'srgb' } );

    const paths: RenderPath[] = [];
    this.depthFirst( program => {
      if ( program instanceof RenderPathProgram && program.path !== null ) {
        paths.push( program.path );
      }
    } );

    const shapes = paths.map( path => {
      const shape = new Shape();
      path.subpaths.forEach( subpath => shape.polygon( subpath ) );
      return shape;
    } );

    for ( let y = 0; y < height; y++ ) {
      for ( let x = 0; x < width; x++ ) {
        const accumulated = new Vector4( 0, 0, 0, 0 );

        for ( let dx = 0; dx < n; dx++ ) {
          for ( let dy = 0; dy < n; dy++ ) {
            // multisample it
            const point = new Vector2( x + ( dx + 0.5 ) / n, y + ( dy + 0.5 ) / n );

            const set = new Set<RenderPath>();

            for ( let p = 0; p < paths.length; p++ ) {
              const path = paths[ p ];
              const shape = shapes[ p ];
              const inside = shape.containsPoint( point );
              if ( inside ) {
                set.add( path );
              }
            }

            const linearPremultiplied = this.evaluate( point, path => set.has( path ) );
            accumulated.add( linearPremultiplied );
          }
        }

        const scaled = accumulated.timesScalar( 1 / ( n * n ) );

        const color = RenderColor.premultipliedLinearToColor( scaled );
        imageData.data[ ( y * width + x ) * 4 ] = color.r;
        imageData.data[ ( y * width + x ) * 4 + 1 ] = color.g;
        imageData.data[ ( y * width + x ) * 4 + 2 ] = color.b;
        imageData.data[ ( y * width + x ) * 4 + 3 ] = color.a * 255;
      }
    }

    return imageData;
  }
}
scenery.register( 'RenderProgram', RenderProgram );

export class RenderBlendCompose extends RenderProgram {
  public constructor(
    public readonly composeType: RenderComposeType,
    public readonly blendType: RenderBlendType,
    public readonly a: RenderProgram,
    public readonly b: RenderProgram
  ) {
    super();
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return other instanceof RenderBlendCompose &&
           this.composeType === other.composeType &&
           this.blendType === other.blendType &&
           this.a.equals( other.a ) &&
           this.b.equals( other.b );
  }

  public override depthFirst( callback: ( program: RenderProgram ) => void ): void {
    this.a.depthFirst( callback );
    this.b.depthFirst( callback );
    callback( this );
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = alwaysTrue ): RenderProgram {
    // a OP b
    const a = this.a.simplify( pathTest );
    const b = this.b.simplify( pathTest );

    const aTransparent = a.isFullyTransparent();
    const aOpaque = a.isFullyOpaque();
    const bTransparent = b.isFullyTransparent();
    const bOpaque = b.isFullyOpaque();

    if ( aTransparent && bTransparent ) {
      return RenderColor.TRANSPARENT;
    }

    // e.g. only A's contribution
    const mixNormal = this.blendType === RenderBlendType.Normal;

    // TODO: review these, there was a bug in atop!
    // over: fa: 1,   fb: 1-a   fa*a: a      fb*b: b(1-a) sum: a + b(1-a)
    // in:   fa: b,   fb: 0     fa*a: ab     fb*b: 0      sum: ab
    // out:  fa: 1-b, fb: 0     fa*a: a(1-b) fb*b: 0      sum: a(1-b)
    // atop: fa: b,   fb: 1-a   fa*a: ab     fb*b: b(1-a) sum: b
    // xor:  fa: 1-b, fb: 1-a   fa*a: a(1-b) fb*b: b(1-a) sum: a(1-b) + b(1-a)
    // plus: fa: 1,   fb: 1     fa*a: a      fb*b: b      sum: a + b
    // ... plusLighter: NOT THIS
    switch( this.composeType ) {
      case RenderComposeType.Over:
        if ( mixNormal && aOpaque ) {
          return a;
        }
        if ( aTransparent ) {
          return b;
        }
        break;
      case RenderComposeType.In:
        if ( aTransparent || bTransparent ) {
          return RenderColor.TRANSPARENT;
        }
        if ( mixNormal && bOpaque ) {
          return a;
        }
        break;
      case RenderComposeType.Out:
        if ( aTransparent || bOpaque ) {
          return RenderColor.TRANSPARENT;
        }
        if ( mixNormal && bTransparent ) {
          return a;
        }
        break;
      case RenderComposeType.Atop:
        if ( bTransparent ) {
          return RenderColor.TRANSPARENT;
        }
        if ( mixNormal && aOpaque && bOpaque ) {
          return a;
        }
        if ( aTransparent ) {
          return b;
        }
        break;
      case RenderComposeType.Xor:
        if ( aOpaque && bOpaque ) {
          return RenderColor.TRANSPARENT;
        }
        if ( aTransparent ) {
          return b;
        }
        if ( bTransparent ) {
          return a;
        }
        break;
      case RenderComposeType.Plus:
      case RenderComposeType.PlusLighter:
        if ( aTransparent ) {
          return b;
        }
        if ( bTransparent ) {
          return a;
        }
        break;
      default:
        break;
    }

    if ( a instanceof RenderColor && b instanceof RenderColor && a.path === null && b.path === null ) {
      return new RenderColor( null, RenderBlendCompose.blendCompose( a.color, b.color, this.composeType, this.blendType ) );
    }

    return new RenderBlendCompose( this.composeType, this.blendType, a, b );
  }

  public override isFullyTransparent(): boolean {
    const aTransparent = this.a.isFullyTransparent();
    const aOpaque = this.a.isFullyOpaque();
    const bTransparent = this.b.isFullyTransparent();
    const bOpaque = this.b.isFullyOpaque();

    if ( aTransparent && bTransparent ) {
      return true;
    }

    // over: fa: 1,   fb: 1-a   fa*a: a      fb*b: b(1-a) sum: a + b(1-a)
    // in:   fa: b,   fb: 0     fa*a: ab     fb*b: 0      sum: ab
    // out:  fa: 1-b, fb: 0     fa*a: a(1-b) fb*b: 0      sum: a(1-b)
    // atop: fa: b,   fb: 1-a   fa*a: ab     fb*b: b(1-a) sum: b
    // xor:  fa: 1-b, fb: 1-a   fa*a: a(1-b) fb*b: b(1-a) sum: a(1-b) + b(1-a)
    // plus: fa: 1,   fb: 1     fa*a: a      fb*b: b      sum: a + b
    switch( this.composeType ) {
      case RenderComposeType.In:
        return aTransparent || bTransparent;
      case RenderComposeType.Out:
        return aTransparent || bOpaque;
      case RenderComposeType.Atop:
        return bTransparent;
      case RenderComposeType.Xor:
        return aOpaque && bOpaque;
      default:
        return false;
    }
  }

  public override isFullyOpaque(): boolean {
    const aTransparent = this.a.isFullyTransparent();
    const aOpaque = this.a.isFullyOpaque();
    const bTransparent = this.b.isFullyTransparent();
    const bOpaque = this.b.isFullyOpaque();

    if ( aTransparent && bTransparent ) {
      return false;
    }

    // over: fa: 1,   fb: 1-a   fa*a: a      fb*b: b(1-a) sum: a + b(1-a)
    // in:   fa: b,   fb: 0     fa*a: ab     fb*b: 0      sum: ab
    // out:  fa: 1-b, fb: 0     fa*a: a(1-b) fb*b: 0      sum: a(1-b)
    // atop: fa: b,   fb: 1-a   fa*a: ab     fb*b: b(1-a) sum: b
    // xor:  fa: 1-b, fb: 1-a   fa*a: a(1-b) fb*b: b(1-a) sum: a(1-b) + b(1-a)
    // plus: fa: 1,   fb: 1     fa*a: a      fb*b: b      sum: a + b
    switch( this.composeType ) {
      case RenderComposeType.Over:
        return aOpaque || bOpaque;
      case RenderComposeType.In:
        return aOpaque && bOpaque;
      case RenderComposeType.Out:
        return aOpaque && bTransparent;
      case RenderComposeType.Atop:
        return bOpaque;
      case RenderComposeType.Xor:
        return ( aOpaque && bTransparent ) || ( aTransparent && bOpaque );
      case RenderComposeType.Plus:
        return aOpaque && bOpaque;
      case RenderComposeType.PlusLighter:
        return aOpaque && bOpaque;
      default:
        return false;
    }
  }

  public override evaluate( point: Vector2, pathTest: ( renderPath: RenderPath ) => boolean = alwaysTrue ): Vector4 {
    const a = this.a.evaluate( point, pathTest );
    const b = this.b.evaluate( point, pathTest );

    return RenderBlendCompose.blendCompose( a, b, this.composeType, this.blendType );
  }

  public override toRecursiveString( indent: string ): string {
    return `${indent}RenderBlendCompose(${RenderComposeType[ this.composeType ]}, ${RenderBlendType[ this.blendType ]}),\n` +
           `${this.a.toRecursiveString( indent + '  ' )},\n` +
           `${this.b.toRecursiveString( indent + '  ' )}`;
  }

  public static blendCompose( a: Vector4, b: Vector4, composeType: RenderComposeType, blendType: RenderBlendType ): Vector4 {
    let blended;
    if ( blendType === RenderBlendType.Normal ) {
      blended = a;
    }
    else {
      // Need to apply blending when not premultiplied
      const a3 = RenderColor.unpremultiply( a ).toVector3();
      const b3 = RenderColor.unpremultiply( b ).toVector3();
      let c3: Vector3;

      switch( blendType ) {
        case RenderBlendType.Multiply: {
          c3 = b3.componentTimes( a3 );
          break;
        }
        case RenderBlendType.Screen: {
          c3 = RenderBlendCompose.screen( b3, a3 );
          break;
        }
        case RenderBlendType.Overlay: {
          c3 = RenderBlendCompose.hardLight( a3, b3 );
          break;
        }
        case RenderBlendType.Darken: {
          c3 = new Vector3(
            Math.min( b3.x, a3.x ),
            Math.min( b3.y, a3.y ),
            Math.min( b3.z, a3.z )
          );
          break;
        }
        case RenderBlendType.Lighten: {
          c3 = new Vector3(
            Math.max( b3.x, a3.x ),
            Math.max( b3.y, a3.y ),
            Math.max( b3.z, a3.z )
          );
          break;
        }
        case RenderBlendType.ColorDodge: {
          c3 = new Vector3(
            RenderBlendCompose.colorDodge( b3.x, a3.x ),
            RenderBlendCompose.colorDodge( b3.y, a3.y ),
            RenderBlendCompose.colorDodge( b3.z, a3.z )
          );
          break;
        }
        case RenderBlendType.ColorBurn: {
          c3 = new Vector3(
            RenderBlendCompose.colorBurn( b3.x, a3.x ),
            RenderBlendCompose.colorBurn( b3.y, a3.y ),
            RenderBlendCompose.colorBurn( b3.z, a3.z )
          );
          break;
        }
        case RenderBlendType.HardLight: {
          c3 = RenderBlendCompose.hardLight( b3, a3 );
          break;
        }
        case RenderBlendType.SoftLight: {
          c3 = RenderBlendCompose.softLight( b3, a3 );
          break;
        }
        case RenderBlendType.Difference: {
          c3 = new Vector3(
            Math.abs( b3.x - a3.x ),
            Math.abs( b3.y - a3.y ),
            Math.abs( b3.z - a3.z )
          );
          break;
        }
        case RenderBlendType.Exclusion: {
          c3 = b3.plus( a3 ).minus( b3.componentTimes( a3 ).timesScalar( 2 ) );
          break;
        }
        case RenderBlendType.Hue: {
          c3 = RenderBlendCompose.setLum( RenderBlendCompose.setSat( a3, RenderBlendCompose.sat( b3 ) ), RenderBlendCompose.lum( b3 ) );
          break;
        }
        case RenderBlendType.Saturation: {
          c3 = RenderBlendCompose.setLum( RenderBlendCompose.setSat( b3, RenderBlendCompose.sat( a3 ) ), RenderBlendCompose.lum( b3 ) );
          break;
        }
        case RenderBlendType.Color: {
          c3 = RenderBlendCompose.setLum( a3, RenderBlendCompose.lum( b3 ) );
          break;
        }
        case RenderBlendType.Luminosity: {
          c3 = RenderBlendCompose.setLum( b3, RenderBlendCompose.lum( a3 ) );
          break;
        }
        default: {
          c3 = a3;
          break;
        }
      }

      blended = RenderColor.premultiply( new Vector4( c3.x, c3.y, c3.z, a.w ) );
    }

    let fa;
    let fb;

    // over: fa: 1,   fb: 1-a   fa*a: a      fb*b: b(1-a) sum: a + b(1-a)
    // in:   fa: b,   fb: 0     fa*a: ab     fb*b: 0      sum: ab
    // out:  fa: 1-b, fb: 0     fa*a: a(1-b) fb*b: 0      sum: a(1-b)
    // atop: fa: b,   fb: 1-a   fa*a: ab     fb*b: b(1-a) sum: b
    // xor:  fa: 1-b, fb: 1-a   fa*a: a(1-b) fb*b: b(1-a) sum: a(1-b) + b(1-a)
    // plus: fa: 1,   fb: 1     fa*a: a      fb*b: b      sum: a + b
    switch( composeType ) {
      case RenderComposeType.Over:
        fa = 1;
        fb = 1 - a.w;
        break;
      case RenderComposeType.In:
        fa = b.w;
        fb = 0;
        break;
      case RenderComposeType.Out:
        fa = 1 - b.w;
        fb = 0;
        break;
      case RenderComposeType.Atop:
        fa = b.w;
        fb = 1 - a.w;
        break;
      case RenderComposeType.Xor:
        fa = 1 - b.w;
        fb = 1 - a.w;
        break;
      case RenderComposeType.Plus:
        fa = 1;
        fb = 1;
        break;
      case RenderComposeType.PlusLighter:
        return new Vector4(
          Math.min( a.x + b.x, 1 ),
          Math.min( a.y + b.y, 1 ),
          Math.min( a.z + b.z, 1 ),
          Math.min( a.w + b.w, 1 )
        );
      default:
        throw new Error( 'unimplemented composeType' );
    }

    return new Vector4(
      // Modes like COMPOSE_PLUS can generate alpha > 1.0, so clamp.
      Math.min( fa * blended.x + fb * b.x, 1 ),
      Math.min( fa * blended.y + fb * b.y, 1 ),
      Math.min( fa * blended.z + fb * b.z, 1 ),
      Math.min( a.w * fa + b.w * fb, 1 )
    );
  }

  public static screen( cb: Vector3, cs: Vector3 ): Vector3 {
    return cb.plus( cs ).minus( cb.componentTimes( cs ) );
  }

  public static colorDodge( cb: number, cs: number ): number {
    if ( cb === 0 ) {
      return 0;
    }
    else if ( cs === 1 ) {
      return 1;
    }
    else {
      return Math.min( 1, cb / ( 1 - cs ) );
    }
  }

  public static colorBurn( cb: number, cs: number ): number {
    if ( cb === 1 ) {
      return 1;
    }
    else if ( cs === 0 ) {
      return 0;
    }
    else {
      return 1 - Math.min( 1, ( 1 - cb ) / cs );
    }
  }

  // TODO: check examples, tests
  public static hardLight( cb: Vector3, cs: Vector3 ): Vector3 {
    const falseOption = RenderBlendCompose.screen( cb, cs.timesScalar( 2 ).minusScalar( 1 ) );
    const trueOption = cb.timesScalar( 2 ).componentTimes( cs );

    return new Vector3(
      cs.x <= 0.5 ? trueOption.x : falseOption.x,
      cs.y <= 0.5 ? trueOption.y : falseOption.y,
      cs.z <= 0.5 ? trueOption.z : falseOption.z
    );
  }

  // TODO: check examples, tests
  public static softLight( cb: Vector3, cs: Vector3 ): Vector3 {
    const d = new Vector3(
      cb.x <= 0.25 ? ( 16 * cb.x - 12 ) * cb.x + 4 : Math.sqrt( cb.x ),
      cb.y <= 0.25 ? ( 16 * cb.y - 12 ) * cb.y + 4 : Math.sqrt( cb.y ),
      cb.z <= 0.25 ? ( 16 * cb.z - 12 ) * cb.z + 4 : Math.sqrt( cb.z )
    );
    return new Vector3(
      cs.x > 0.5 ? cb.x + ( 2 * cs.x - 1 ) * ( d.x - cb.x ) : cb.x - ( 1 - 2 * cs.x ) * cb.x * ( 1 - cb.x ),
      cs.y > 0.5 ? cb.y + ( 2 * cs.y - 1 ) * ( d.y - cb.y ) : cb.y - ( 1 - 2 * cs.y ) * cb.y * ( 1 - cb.y ),
      cs.z > 0.5 ? cb.z + ( 2 * cs.z - 1 ) * ( d.z - cb.z ) : cb.z - ( 1 - 2 * cs.z ) * cb.z * ( 1 - cb.z )
    );
  }

  // TODO: check examples, tests
  public static sat( c: Vector3 ): number {
    return Math.max( c.x, Math.max( c.y, c.z ) ) - Math.min( c.x, Math.min( c.y, c.z ) );
  }

  // TODO: check examples, tests
  public static lum( c: Vector3 ): number {
    return 0.3 * c.x + 0.59 * c.y + 0.11 * c.z;
  }

  // TODO: check examples, tests
  public static clipColor( c: Vector3 ): Vector3 {
    const lScalar = RenderBlendCompose.lum( c );
    const l = new Vector3( lScalar, lScalar, lScalar );
    const n = Math.min( c.x, c.y, c.z );
    const x = Math.max( c.x, c.y, c.z );
    if ( n < 0 ) {
      c = l.plus( c.minus( l ).componentTimes( l ).dividedScalar( lScalar - n ) );
    }
    if ( x > 1 ) {
      c = l.plus( c.minus( l ).timesScalar( ( 1 - lScalar ) / ( x - lScalar ) ) );
    }
    return c;
  }

  public static setLum( c: Vector3, l: number ): Vector3 {
    return RenderBlendCompose.clipColor( c.plusScalar( l - RenderBlendCompose.lum( c ) ) );
  }

  public static setSatInner( c: Vector3, s: number ): void {
    if ( c.z > c.x ) {
      c.y = ( ( c.y - c.x ) * s ) / ( c.z - c.x );
      c.z = s;
    }
    else {
      c.y = 0;
      c.z = 0;
    }
    c.x = 0;
  }

  public static setSat( c: Vector3, s: number ): Vector3 {
    // TODO: this is horribly inefficient and wonky

    if ( c.x <= c.y ) {
      if ( c.y <= c.z ) {
        RenderBlendCompose.setSatInner( c, s );
        return c;
      }
      else {
        if ( c.x <= c.z ) {
          const v = new Vector3( c.x, c.z, c.y );
          RenderBlendCompose.setSatInner( v, s );
          return new Vector3( v.x, v.z, v.y );
        }
        else {
          const v = new Vector3( c.z, c.x, c.y );
          RenderBlendCompose.setSatInner( v, s );
          return new Vector3( v.y, v.z, v.x );
        }
      }
    }
    else {
      if ( c.x <= c.z ) {
        const v = new Vector3( c.y, c.x, c.z );
        RenderBlendCompose.setSatInner( v, s );
        return new Vector3( v.y, v.x, v.z );
      }
      else {
        if ( c.y <= c.z ) {
          const v = new Vector3( c.y, c.z, c.x );
          RenderBlendCompose.setSatInner( v, s );
          return new Vector3( v.z, v.x, v.y );
        }
        else {
          const v = new Vector3( c.z, c.y, c.x );
          RenderBlendCompose.setSatInner( v, s );
          return new Vector3( v.z, v.y, v.x );
        }
      }
    }
  }
}
scenery.register( 'RenderBlendCompose', RenderBlendCompose );

let globalPathId = 0;

export class RenderPath {

  public readonly id = globalPathId++;

  public constructor( public readonly fillRule: FillRule, public readonly subpaths: Vector2[][] ) {}
}
scenery.register( 'RenderPath', RenderPath );

export abstract class RenderPathProgram extends RenderProgram {
  protected constructor( public readonly path: RenderPath | null ) {
    super();
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return other instanceof RenderPathProgram &&
           this.path === other.path;
  }

  public isInPath( pathTest: ( renderPath: RenderPath ) => boolean ): boolean {
    return !this.path || pathTest( this.path );
  }
}
scenery.register( 'RenderPathProgram', RenderPathProgram );

export class RenderFilter extends RenderPathProgram {
  public constructor(
    path: RenderPath | null,
    public readonly program: RenderProgram,
    public readonly matrix: Matrix4
  ) {
    super( path );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return super.equals( other ) &&
           other instanceof RenderFilter &&
           this.program.equals( other.program ) &&
           this.matrix.equals( other.matrix );
  }

  public override depthFirst( callback: ( program: RenderProgram ) => void ): void {
    this.program.depthFirst( callback );
    callback( this );
  }

  // TODO: inspect matrix to see when it will maintain transparency!
  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = alwaysTrue ): RenderProgram {
    const program = this.program.simplify( pathTest );

    if ( this.isInPath( pathTest ) ) {
      if ( program instanceof RenderColor ) {
        return new RenderColor( null, RenderColor.premultiply( this.matrix.timesVector4( RenderColor.unpremultiply( program.color ) ) ) );
      }
      else {
        return new RenderFilter( this.path, program, this.matrix );
      }
    }
    else {
      return program;
    }
  }

  public override isFullyTransparent(): boolean {
    // TODO: color matrix check. Homogeneous?
    return false;
  }

  public override isFullyOpaque(): boolean {
    // TODO: color matrix check. Homogeneous?
    return false;
  }

  public override evaluate( point: Vector2, pathTest: ( renderPath: RenderPath ) => boolean = alwaysTrue ): Vector4 {
    const source = this.program.evaluate( point, pathTest );

    if ( this.isInPath( pathTest ) ) {
      return RenderColor.premultiply( this.matrix.timesVector4( RenderColor.unpremultiply( source ) ) );
    }
    else {
      return source;
    }
  }

  public override toRecursiveString( indent: string ): string {
    return `${indent}RenderFilter (${this.path ? this.path.id : 'null'})\n` +
           `${this.program.toRecursiveString( indent + '  ' )}`;
  }
}
scenery.register( 'RenderFilter', RenderFilter );

export class RenderAlpha extends RenderPathProgram {
  public constructor(
    path: RenderPath | null,
    public readonly program: RenderProgram,
    public readonly alpha: number
  ) {
    super( path );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return super.equals( other ) &&
           other instanceof RenderAlpha &&
           this.program.equals( other.program ) &&
           this.alpha === other.alpha;
  }

  public override depthFirst( callback: ( program: RenderProgram ) => void ): void {
    this.program.depthFirst( callback );
    callback( this );
  }

  public override isFullyTransparent(): boolean {
    if ( this.path ) {
      return this.program.isFullyTransparent();
    }
    else {
      return this.alpha === 0 || this.program.isFullyTransparent();
    }
  }

  public override isFullyOpaque(): boolean {
    return this.alpha === 1 && this.program.isFullyOpaque();
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = alwaysTrue ): RenderProgram {
    const program = this.program.simplify( pathTest );
    if ( program.isFullyTransparent() || this.alpha === 0 ) {
      return RenderColor.TRANSPARENT;
    }

    // No difference inside-outside
    if ( this.alpha === 1 || !this.isInPath( pathTest ) ) {
      return program;
    }

    // Now we're "inside" our path
    if ( program instanceof RenderColor ) {
      return new RenderColor( null, program.color.timesScalar( this.alpha ) );
    }
    else {
      return new RenderAlpha( null, program, this.alpha );
    }
  }

  public override evaluate( point: Vector2, pathTest: ( renderPath: RenderPath ) => boolean = alwaysTrue ): Vector4 {
    const source = this.program.evaluate( point, pathTest );

    if ( this.isInPath( pathTest ) ) {
      return source.timesScalar( this.alpha );
    }
    else {
      return source;
    }
  }

  public override toRecursiveString( indent: string ): string {
    return `${indent}RenderAlpha (${this.path ? this.path.id : 'null'}, alpha:${this.alpha})\n` +
           `${this.program.toRecursiveString( indent + '  ' )}`;
  }
}
scenery.register( 'RenderAlpha', RenderAlpha );

// TODO: consider transforms as a node itself? Meh probably excessive?

export class RenderColor extends RenderPathProgram {
  public constructor( path: RenderPath | null, public color: Vector4 ) {
    super( path );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return super.equals( other ) &&
           other instanceof RenderColor &&
           this.color.equals( other.color );
  }

  public override isFullyTransparent(): boolean {
    return this.color.w === 0;
  }

  public override isFullyOpaque(): boolean {
    return this.path === null && this.color.w === 1;
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = alwaysTrue ): RenderProgram {
    if ( this.isInPath( pathTest ) ) {
      return new RenderColor( null, this.color );
    }
    else {
      return RenderColor.TRANSPARENT;
    }
  }

  public override evaluate( point: Vector2, pathTest: ( renderPath: RenderPath ) => boolean = alwaysTrue ): Vector4 {
    if ( this.isInPath( pathTest ) ) {
      return this.color;
    }
    else {
      return Vector4.ZERO;
    }
  }

  public override toRecursiveString( indent: string ): string {
    return `${indent}RenderColor (${this.path ? this.path.id : 'null'}, color:${this.color.toString()})`;
  }

  public static fromColor( path: RenderPath | null, color: Color ): RenderColor {
    return new RenderColor( path, RenderColor.colorToPremultipliedLinear( color ) );
  }

  public static colorToPremultipliedLinear( color: Color ): Vector4 {
    // https://entropymine.com/imageworsener/srgbformula/
    // sRGB to Linear
    // 0 ≤ S ≤ 0.0404482362771082 : L = S/12.92
    // 0.0404482362771082 < S ≤ 1 : L = ((S+0.055)/1.055)^2.4

    // Linear to sRGB
    // 0 ≤ L ≤ 0.00313066844250063 : S = L×12.92
    // 0.00313066844250063 < L ≤ 1 : S = 1.055×L^1/2.4 − 0.055

    const sRGB = new Vector4(
      color.red / 255,
      color.green / 255,
      color.blue / 255,
      color.alpha
    );

    return RenderColor.premultiply( RenderColor.sRGBToLinear( sRGB ) );
  }

  public static premultipliedLinearToColor( premultiplied: Vector4 ): Color {
    const sRGB = RenderColor.linearToSRGB( RenderColor.unpremultiply( premultiplied ) );

    return new Color(
      sRGB.x * 255,
      sRGB.y * 255,
      sRGB.z * 255,
      sRGB.w
    );
  }

  public static sRGBToLinear( sRGB: Vector4 ): Vector4 {
    return new Vector4(
      sRGB.x <= 0.0404482362771082 ? sRGB.x / 12.92 : Math.pow( ( sRGB.x + 0.055 ) / 1.055, 2.4 ),
      sRGB.y <= 0.0404482362771082 ? sRGB.y / 12.92 : Math.pow( ( sRGB.y + 0.055 ) / 1.055, 2.4 ),
      sRGB.z <= 0.0404482362771082 ? sRGB.z / 12.92 : Math.pow( ( sRGB.z + 0.055 ) / 1.055, 2.4 ),
      sRGB.w
    );
  }

  public static linearToSRGB( linear: Vector4 ): Vector4 {
    return new Vector4(
      linear.x <= 0.00313066844250063 ? linear.x * 12.92 : 1.055 * Math.pow( linear.x, 1 / 2.4 ) - 0.055,
      linear.y <= 0.00313066844250063 ? linear.y * 12.92 : 1.055 * Math.pow( linear.y, 1 / 2.4 ) - 0.055,
      linear.z <= 0.00313066844250063 ? linear.z * 12.92 : 1.055 * Math.pow( linear.z, 1 / 2.4 ) - 0.055,
      linear.w
    );
  }

  public static premultiply( color: Vector4 ): Vector4 {
    return new Vector4(
      color.x * color.w,
      color.y * color.w,
      color.z * color.w,
      color.w
    );
  }

  public static unpremultiply( color: Vector4 ): Vector4 {
    return color.w === 0 ? Vector4.ZERO : new Vector4(
      color.x / color.w,
      color.y / color.w,
      color.z / color.w,
      color.w
    );
  }

  public static readonly TRANSPARENT = new RenderColor( null, Vector4.ZERO );
}
scenery.register( 'RenderColor', RenderColor );

type RenderImageable = {
  width: number;
  height: number;

  // TODO: derivatives for filtering? (don't really need that right?)

  // TODO: sampling of things, actually have methods that get samples (in any color space)
  evaluate: ( point: Vector2 ) => Color;
};

export class RenderImage extends RenderPathProgram {
  public constructor(
    path: RenderPath | null,
    public readonly transform: Matrix3,
    public readonly image: RenderImageable,
    public readonly extendX: RenderExtend,
    public readonly extendY: RenderExtend
  ) {
    super( path );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return super.equals( other ) &&
      other instanceof RenderImage &&
      this.transform.equals( other.transform ) &&
      this.image === other.image &&
      this.extendX === other.extendX &&
      this.extendY === other.extendY;
  }

  public override isFullyTransparent(): boolean {
    return false;
  }

  public override isFullyOpaque(): boolean {
    return false;
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = alwaysTrue ): RenderProgram {
    if ( this.isInPath( pathTest ) ) {
      return new RenderImage( null, this.transform, this.image, this.extendX, this.extendY );
    }
    else {
      return RenderColor.TRANSPARENT;
    }
  }

  public override evaluate( point: Vector2, pathTest: ( renderPath: RenderPath ) => boolean = alwaysTrue ): Vector4 {
    if ( !this.isInPath( pathTest ) ) {
      return Vector4.ZERO;
    }

    const localPoint = this.transform.inverted().timesVector2( point );
    const tx = localPoint.x / this.image.width;
    const ty = localPoint.y / this.image.height;
    const mappedX = RenderImage.extend( this.extendX, tx );
    const mappedY = RenderImage.extend( this.extendY, ty );

    const color = this.image.evaluate( new Vector2( mappedX * this.image.width, mappedY * this.image.height ) );

    return RenderColor.colorToPremultipliedLinear( color );
  }

  public override toRecursiveString( indent: string ): string {
    return `${indent}RenderImage (${this.path ? this.path.id : 'null'})`;
  }

  public static extend( extend: RenderExtend, t: number ): number {
    switch( extend ) {
      case RenderExtend.Pad:
        return Utils.clamp( t, 0, 1 );
      case RenderExtend.Repeat:
        return t - Math.floor( t );
      case RenderExtend.Reflect:
        return Math.abs( t - 2.0 * Utils.roundSymmetric( 0.5 * t ) );
        // return ( Math.floor( t ) % 2 === 0 ? t : 1 - t ) - Math.floor( t );
      default:
        throw new Error( 'Unknown RenderExtend' );
    }
  }

  // Integer version of extend_mode.
  // Given size=4, provide the following patterns:
  //
  // input:  -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
  //
  // pad:     0,  0,  0,  0,  0,  0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3
  // repeat:  2,  3,  0,  1,  2,  3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1
  // reflect: 2,  3,  3,  2,  1,  0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1
  public static extendInteger( i: number, size: number, extend: RenderExtend ): number {
    switch( extend ) {
      case RenderExtend.Pad: {
        return Utils.clamp( i, 0, size - 1 );
      }
      case RenderExtend.Repeat: {
        if ( i >= 0 ) {
          return i % size;
        }
        else {
          return size - ( ( -i - 1 ) % size ) - 1;
        }
      }
      case RenderExtend.Reflect: {
        // easier to convert both to positive (with a repeat offset)
        const positiveI = i < 0 ? -i - 1 : i;

        const section = positiveI % ( size * 2 );
        if ( section < size ) {
          return section;
        }
        else {
          return 2 * size - section - 1;
        }
      }
      default: {
        throw new Error( 'Unknown RenderExtend' );
      }
    }
  }
}
scenery.register( 'RenderImage', RenderImage );

export class RenderGradientStop {
  public constructor( public readonly ratio: number, public readonly program: RenderProgram ) {}

  public static evaluate( point: Vector2, stops: RenderGradientStop[], t: number, pathTest: ( renderPath: RenderPath ) => boolean = alwaysTrue ): Vector4 {
    let i = -1;
    while ( i < stops.length - 1 && stops[ i + 1 ].ratio < t ) {
      i++;
    }
    if ( i === -1 ) {
      return stops[ 0 ].program.evaluate( point, pathTest );
    }
    else if ( i === stops.length - 1 ) {
      return stops[ i ].program.evaluate( point, pathTest );
    }
    else {
      const before = stops[ i ];
      const after = stops[ i + 1 ];
      const ratio = ( t - before.ratio ) / ( after.ratio - before.ratio );

      const beforeColor = before.program.evaluate( point, pathTest );
      const afterColor = after.program.evaluate( point, pathTest );

      const minusRatio = 1 - ratio;

      // TODO: reduce allocation
      return new Vector4(
        beforeColor.x * minusRatio + afterColor.x * ratio,
        beforeColor.y * minusRatio + afterColor.y * ratio,
        beforeColor.z * minusRatio + afterColor.z * ratio,
        beforeColor.w * minusRatio + afterColor.w * ratio
      );
    }
  }
}
scenery.register( 'RenderGradientStop', RenderGradientStop );

const scratchLinearGradientVector0 = new Vector2( 0, 0 );

export class RenderLinearGradient extends RenderPathProgram {

  private readonly inverseTransform: Matrix3;
  private readonly isIdentity: boolean;
  private readonly gradDelta: Vector2;

  public constructor(
    path: RenderPath | null,
    public readonly transform: Matrix3,
    public readonly start: Vector2,
    public readonly end: Vector2,
    public readonly stops: RenderGradientStop[], // should be sorted!!
    public readonly extend: RenderExtend
  ) {
    super( path );

    this.inverseTransform = transform.inverted();
    this.isIdentity = transform.isIdentity();
    this.gradDelta = end.minus( start );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return super.equals( other ) &&
      other instanceof RenderLinearGradient &&
      this.transform.equals( other.transform ) &&
      this.start.equals( other.start ) &&
      this.end.equals( other.end ) &&
      this.stops.length === other.stops.length &&
      // TODO perf
      this.stops.every( ( stop, i ) => stop.ratio === other.stops[ i ].ratio && stop.program.equals( other.stops[ i ].program ) ) &&
      this.extend === other.extend;
  }

  public override depthFirst( callback: ( program: RenderProgram ) => void ): void {
    this.stops.forEach( stop => stop.program.depthFirst( callback ) );
    callback( this );
  }

  public override isFullyTransparent(): boolean {
    return this.stops.every( stop => stop.program.isFullyTransparent() );
  }

  public override isFullyOpaque(): boolean {
    return this.path === null && this.stops.every( stop => stop.program.isFullyOpaque() );
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = alwaysTrue ): RenderProgram {
    const simplifiedColorStops = this.stops.map( stop => new RenderGradientStop( stop.ratio, stop.program.simplify( pathTest ) ) );

    if ( simplifiedColorStops.every( stop => stop.program.isFullyTransparent() ) ) {
      return RenderColor.TRANSPARENT;
    }

    if ( this.isInPath( pathTest ) ) {
      return new RenderLinearGradient( null, this.transform, this.start, this.end, simplifiedColorStops, this.extend );
    }
    else {
      return RenderColor.TRANSPARENT;
    }
  }

  public override evaluate( point: Vector2, pathTest: ( renderPath: RenderPath ) => boolean = alwaysTrue ): Vector4 {
    if ( !this.isInPath( pathTest ) ) {
      return Vector4.ZERO;
    }

    const localPoint = scratchLinearGradientVector0.set( point );
    if ( !this.isIdentity ) {
      this.inverseTransform.multiplyVector2( localPoint );
    }

    const localDelta = localPoint.subtract( this.start ); // MUTABLE, changes localPoint
    const gradDelta = this.gradDelta;

    const t = gradDelta.magnitude > 0 ? localDelta.dot( gradDelta ) / gradDelta.dot( gradDelta ) : 0;
    const mappedT = RenderImage.extend( this.extend, t );

    return RenderGradientStop.evaluate( point, this.stops, mappedT, pathTest );
  }

  public override toRecursiveString( indent: string ): string {
    return `${indent}RenderLinearGradient (${this.path ? this.path.id : 'null'})`;
  }
}
scenery.register( 'RenderLinearGradient', RenderLinearGradient );

// TODO: gradients/linears in... sRGB space? or perceptual? linear blends don't seem ideal
const scratchLinearBlendVector = new Vector2( 0, 0 );
export class RenderLinearBlend extends RenderPathProgram {

  public constructor(
    path: RenderPath | null,
    public readonly scaledNormal: Vector2,
    public readonly offset: number,
    public readonly zero: RenderProgram,
    public readonly one: RenderProgram
  ) {
    super( path );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return super.equals( other ) &&
      other instanceof RenderLinearBlend &&
      this.scaledNormal.equals( other.scaledNormal ) &&
      this.offset === other.offset &&
      this.zero.equals( other.zero ) &&
      this.one.equals( other.one );
  }

  public override depthFirst( callback: ( program: RenderProgram ) => void ): void {
    this.zero.depthFirst( callback );
    this.one.depthFirst( callback );
    callback( this );
  }

  public override isFullyTransparent(): boolean {
    return this.zero.isFullyTransparent() && this.one.isFullyTransparent();
  }

  public override isFullyOpaque(): boolean {
    return this.path === null && this.zero.isFullyOpaque() && this.one.isFullyOpaque();
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = alwaysTrue ): RenderProgram {
    const zero = this.zero.simplify( pathTest );
    const one = this.one.simplify( pathTest );

    if ( zero.isFullyTransparent() && one.isFullyTransparent() ) {
      return RenderColor.TRANSPARENT;
    }

    if ( this.isInPath( pathTest ) ) {
      return new RenderLinearBlend( null, this.scaledNormal, this.offset, zero, one );
    }
    else {
      return RenderColor.TRANSPARENT;
    }
  }

  public override evaluate( point: Vector2, pathTest: ( renderPath: RenderPath ) => boolean = alwaysTrue ): Vector4 {
    if ( !this.isInPath( pathTest ) ) {
      return Vector4.ZERO;
    }

    const localPoint = scratchLinearBlendVector.set( point );

    const t = this.scaledNormal.dot( localPoint ) - this.offset;

    if ( t <= 0 ) {
      return this.zero.evaluate( point, pathTest );
    }
    else if ( t >= 1 ) {
      return this.one.evaluate( point, pathTest );
    }
    else {
      return this.zero.evaluate( point, pathTest ).timesScalar( 1 - t ).plus( this.one.evaluate( point, pathTest ).timesScalar( t ) );
    }
  }

  public override toRecursiveString( indent: string ): string {
    return `${indent}RenderLinearGradient (${this.path ? this.path.id : 'null'})`;
  }
}
scenery.register( 'RenderLinearBlend', RenderLinearBlend );

export enum RadialGradientType {
  Circular = 1,
  Strip = 2,
  FocalOnCircle = 3,
  Cone = 4
}

const fromPoly2 = ( p0: Vector2, p1: Vector2 ): Matrix3 => {
  return Matrix3.affine(
    p1.y - p0.y, p1.x - p0.x, p0.x,
    p0.x - p1.x, p1.y - p0.y, p0.y
  );
  // TODO: remove comments once tested
  // return Transform(
  //   vec4(p1.y - p0.y, p0.x - p1.x, p1.x - p0.x, p1.y - p0.y),
  //   vec2(p0.x, p0.y)
  // );
};

const twoPointToUnitLine = ( p0: Vector2, p1: Vector2 ): Matrix3 => {
  return fromPoly2( Vector2.ZERO, Vector2.X_UNIT ).timesMatrix( fromPoly2( p0, p1 ).inverted() );
};

class RadialGradientLogic {

  private readonly xform: Matrix3;
  private readonly focal_x: number;
  private readonly radius: number;
  private readonly kind: RadialGradientType;
  private readonly isSwapped: boolean;

  public constructor( public readonly radialGradient: RenderRadialGradient ) {
    // Two-point conical gradient based on Vello, based on https://skia.org/docs/dev/design/conical/
    let p0 = radialGradient.start;
    let p1 = radialGradient.end;
    let r0 = radialGradient.startRadius;
    let r1 = radialGradient.endRadius;

    const GRADIENT_EPSILON = 1 / ( 1 << 12 );
    const userToGradient = radialGradient.transform.inverted();

    // Output variables
    let xform: Matrix3 | null = null;
    let focal_x = 0;
    let radius = 0;
    let kind: RadialGradientType;
    let isSwapped = false;

    if ( Math.abs( r0 - r1 ) <= GRADIENT_EPSILON ) {
      // When the radii are the same, emit a strip gradient
      kind = RadialGradientType.Strip;
      const scaled = r0 / p0.distance( p1 ); // TODO: how to handle div by zero?
      xform = twoPointToUnitLine( p0, p1 ).timesMatrix( userToGradient );
      radius = scaled * scaled;
    }
    else {
      // Assume a two point conical gradient unless the centers
      // are equal.
      kind = RadialGradientType.Cone;
      if ( p0.equals( p1 ) ) {
        kind = RadialGradientType.Circular;
        // Nudge p0 a bit to avoid denormals.
        p0.addScalar( GRADIENT_EPSILON );
      }
      if ( r1 === 0 ) {
        // If r1 === 0, swap the points and radii
        isSwapped = true;
        const tmp_p = p0;
        p0 = p1;
        p1 = tmp_p;
        const tmp_r = r0;
        r0 = r1;
        r1 = tmp_r;
      }
      focal_x = r0 / ( r0 - r1 );
      const cf = p0.timesScalar( 1 - focal_x ).add( p1.timesScalar( focal_x ) );
      radius = r1 / cf.distance( p1 );
      const user_to_unit_line = twoPointToUnitLine( cf, p1 ).timesMatrix( userToGradient );
      let user_to_scaled = user_to_unit_line;
      // When r === 1, focal point is on circle
      if ( Math.abs( radius - 1 ) <= GRADIENT_EPSILON ) {
        kind = RadialGradientType.FocalOnCircle;
        const scale = 0.5 * Math.abs( 1 - focal_x );
        user_to_scaled = Matrix3.scaling( scale ).timesMatrix( user_to_unit_line );
      }
      else {
        const a = radius * radius - 1;
        const scale_ratio = Math.abs( 1 - focal_x ) / a;
        const scale_x = radius * scale_ratio;
        const scale_y = Math.sqrt( Math.abs( a ) ) * scale_ratio;
        user_to_scaled = Matrix3.scaling( scale_x, scale_y ).timesMatrix( user_to_unit_line );
      }
      xform = user_to_scaled;
    }

    this.xform = xform;
    this.focal_x = focal_x;
    this.radius = radius;
    this.kind = kind;
    this.isSwapped = isSwapped;
  }

  public evaluate( point: Vector2, pathTest: ( renderPath: RenderPath ) => boolean = alwaysTrue ): Vector4 {
    const focal_x = this.focal_x;
    const radius = this.radius;
    const kind = this.kind;
    const is_swapped = this.isSwapped;

    // TODO: remove comments once tested
    const is_strip = kind === RadialGradientType.Strip;
    const is_circular = kind === RadialGradientType.Circular;
    const is_focal_on_circle = kind === RadialGradientType.FocalOnCircle;
    const r1_recip = is_circular ? 0 : 1 / radius;
    // let r1_recip = select(1 / radius, 0, is_circular);
    const less_scale = is_swapped || ( 1 - focal_x ) < 0 ? -1 : 1;
    // let less_scale = select(1, -1, is_swapped || (1 - focal_x) < 0);
    const t_sign = Math.sign( 1 - focal_x );

    // Pixel-specifics
    const local_xy = this.xform.timesVector2( point );
    const x = local_xy.x;
    const y = local_xy.y;
    const xx = x * x;
    const yy = y * y;
    let t = 0;
    let is_valid = true;
    if ( is_strip ) {
      const a = radius - yy;
      t = Math.sqrt( a ) + x;
      is_valid = a >= 0;
    }
    else if ( is_focal_on_circle ) {
      t = ( xx + yy ) / x;
      is_valid = t >= 0 && x !== 0;
    }
    else if ( radius > 1 ) {
      t = Math.sqrt( xx + yy ) - x * r1_recip;
    }
    else { // radius < 1
      const a = xx - yy;
      t = less_scale * Math.sqrt( a ) - x * r1_recip;
      is_valid = a >= 0 && t >= 0;
    }
    if ( is_valid ) {
      t = RenderImage.extend( this.radialGradient.extend, focal_x + t_sign * t );
      if ( is_swapped ) {
        t = 1 - t;
      }

      return RenderGradientStop.evaluate( point, this.radialGradient.stops, t, pathTest );
    }
    else {
      // Invalid is a checkerboard red/yellow
      return ( Utils.roundSymmetric( point.x ) + Utils.roundSymmetric( point.y ) ) % 2 === 0 ? new Vector4( 1, 0, 0, 1 ) : new Vector4( 1, 1, 0, 1 );
    }
  }
}

export class RenderRadialGradient extends RenderPathProgram {

  private logic: RadialGradientLogic | null = null;

  public constructor(
    path: RenderPath | null,
    public readonly transform: Matrix3,
    public readonly start: Vector2,
    public readonly startRadius: number,
    public readonly end: Vector2,
    public readonly endRadius: number,
    public readonly stops: RenderGradientStop[], // should be sorted!!
    public readonly extend: RenderExtend
  ) {
    super( path );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return super.equals( other ) &&
      other instanceof RenderRadialGradient &&
      this.transform.equals( other.transform ) &&
      this.start.equals( other.start ) &&
      this.startRadius === other.startRadius &&
      this.end.equals( other.end ) &&
      this.endRadius === other.endRadius &&
      this.stops.length === other.stops.length &&
      // TODO perf
      this.stops.every( ( stop, i ) => stop.ratio === other.stops[ i ].ratio && stop.program.equals( other.stops[ i ].program ) ) &&
      this.extend === other.extend;
  }

  public override depthFirst( callback: ( program: RenderProgram ) => void ): void {
    this.stops.forEach( stop => stop.program.depthFirst( callback ) );
    callback( this );
  }

  public override isFullyTransparent(): boolean {
    return this.stops.every( stop => stop.program.isFullyTransparent() );
  }

  public override isFullyOpaque(): boolean {
    return this.path === null && this.stops.every( stop => stop.program.isFullyOpaque() );
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = alwaysTrue ): RenderProgram {
    const simplifiedColorStops = this.stops.map( stop => new RenderGradientStop( stop.ratio, stop.program.simplify( pathTest ) ) );

    if ( simplifiedColorStops.every( stop => stop.program.isFullyTransparent() ) ) {
      return RenderColor.TRANSPARENT;
    }

    if ( this.isInPath( pathTest ) ) {
      return new RenderRadialGradient( null, this.transform, this.start, this.startRadius, this.end, this.endRadius, simplifiedColorStops, this.extend );
    }
    else {
      return RenderColor.TRANSPARENT;
    }
  }

  public override evaluate( point: Vector2, pathTest: ( renderPath: RenderPath ) => boolean = alwaysTrue ): Vector4 {
    if ( this.logic === null ) {
      this.logic = new RadialGradientLogic( this );
    }

    return this.logic.evaluate( point, pathTest );
  }

  public override toRecursiveString( indent: string ): string {
    return `${indent}RenderRadialGradient (${this.path ? this.path.id : 'null'})`;
  }
}
scenery.register( 'RenderRadialGradient', RenderRadialGradient );
